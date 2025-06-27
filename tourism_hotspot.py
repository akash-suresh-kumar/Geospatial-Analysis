import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point
from shapely.ops import unary_union
import os
import warnings
warnings.filterwarnings('ignore')

class TourismHotspotAnalyzer:
    def __init__(self, target_crs='EPSG:4326'):
        self.target_crs = target_crs
        self.merged_data = None
        self.clusters = None
        self.hotspot_polygons = None
        
    def load_vector_data(self, file_paths, data_dir):
        gdfs = []
        
        for file_path in file_paths:
            try:
                full_path = os.path.join(data_dir, file_path) if not os.path.isabs(file_path) else file_path
                gdf = gpd.read_file(full_path)
                
                # Convert to points if not already
                if gdf.geometry.geom_type.iloc[0] not in ['Point', 'MultiPoint']:
                    gdf.geometry = gdf.geometry.centroid
                
                if gdf.crs != self.target_crs:
                    gdf = gdf.to_crs(self.target_crs)
                
                gdf['source'] = os.path.basename(full_path)
                gdfs.append(gdf)
                print(f"Loaded: {os.path.basename(full_path)} ({len(gdf)} points)")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if gdfs:
            self.merged_data = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
            self.merged_data.crs = self.target_crs
            print(f"Total points loaded: {len(self.merged_data)}")
        else:
            self.merged_data = gpd.GeoDataFrame()
        
        return self.merged_data
    
    def perform_clustering(self, eps=0.01, min_samples=3):
        if self.merged_data is None or len(self.merged_data) == 0:
            return None
            
        coords = np.array([[point.x, point.y] for point in self.merged_data.geometry])
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(coords)
        
        self.merged_data['cluster'] = cluster_labels
        self.clusters = self.merged_data[self.merged_data['cluster'] != -1].copy()
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"Found {n_clusters} clusters")
        return cluster_labels
    
    def create_hotspot_polygons(self, buffer_distance=0.01):
        if self.clusters is None or len(self.clusters) == 0:
            return None
            
        hotspot_polygons = []
        for cluster_id in self.clusters['cluster'].unique():
            cluster_points = self.clusters[self.clusters['cluster'] == cluster_id]
            points = [Point(row.geometry.x, row.geometry.y) for _, row in cluster_points.iterrows()]
            
            if len(points) >= 3:
                hull = unary_union(points).convex_hull.buffer(buffer_distance)
            else:
                hull = unary_union([p.buffer(buffer_distance) for p in points])
            
            hotspot_polygons.append({
                'cluster_id': cluster_id,
                'num_points': len(cluster_points),
                'geometry': hull
            })
        
        self.hotspot_polygons = gpd.GeoDataFrame(hotspot_polygons, crs=self.target_crs)
        return self.hotspot_polygons
    
    def generate_heatmap_raster(self, output_path, resolution=0.01):
        if self.merged_data is None or len(self.merged_data) == 0:
            return
            
        coords = np.array([[point.x, point.y] for point in self.merged_data.geometry])
        
        buffer = 0.1
        xmin, ymin = coords.min(axis=0) - buffer
        xmax, ymax = coords.max(axis=0) + buffer
        
        width = int((xmax - xmin) / resolution)
        height = int((ymax - ymin) / resolution)
        
        x_range = np.linspace(xmin, xmax, width)
        y_range = np.linspace(ymin, ymax, height)
        X, Y = np.meshgrid(x_range, y_range)
        positions = np.vstack([X.ravel(), Y.ravel()])
        
        kde = gaussian_kde(coords.T)
        density = kde(positions).reshape(X.shape)
        density = np.flipud(density)
        
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
        
        with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, 
                          count=1, dtype=density.dtype, crs=self.target_crs, transform=transform) as dst:
            dst.write(density, 1)
        
        print(f"Heatmap saved: {output_path}")
    
    def run_analysis(self, file_paths, data_dir, output_dir, eps=0.01, min_samples=3):
        os.makedirs(output_dir, exist_ok=True)
        
        self.load_vector_data(file_paths, data_dir)
        
        if self.merged_data is None or len(self.merged_data) == 0:
            print("No valid data loaded.")
            return None
            
        self.perform_clustering(eps=eps, min_samples=min_samples)
        self.create_hotspot_polygons()
        
        # Save outputs
        try:
            self.generate_heatmap_raster(os.path.join(output_dir, 'tourism_heatmap.tiff'))
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            
        try:
            if self.hotspot_polygons is not None and len(self.hotspot_polygons) > 0:
                self.hotspot_polygons.to_file(os.path.join(output_dir, 'tourism_hotspots.geojson'))
        except Exception as e:
            print(f"Error saving hotspots: {e}")
            
        try:
            if self.clusters is not None and len(self.clusters) > 0:
                self.clusters.to_file(os.path.join(output_dir, 'clustered_points.geojson'))
        except Exception as e:
            print(f"Error saving clusters: {e}")
        
        print(f"Tourism analysis complete! Output: {output_dir}")
        
        return {
            'total_points': len(self.merged_data),
            'clustered_points': len(self.clusters) if self.clusters is not None else 0,
            'hotspot_polygons': len(self.hotspot_polygons) if self.hotspot_polygons is not None else 0
        }