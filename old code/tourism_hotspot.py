import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
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
        """
        Initialize TourismHotspotAnalyzer object.

        Parameters:
        target_crs : str, optional
            Target coordinate reference system (CRS) of the data. Defaults to 'EPSG:4326' (WGS84 lat/lon).

        Attributes:
        target_crs : str
            Target coordinate reference system (CRS) of the data.
        merged_data : geopandas.GeoDataFrame or None
            Merged GeoDataFrame of all data loaded from files.
        clusters : geopandas.GeoDataFrame or None
            GeoDataFrame of tourism hotspot clusters.
        hotspot_polygons : geopandas.GeoDataFrame or None
            GeoDataFrame of tourism hotspot polygons.
        """
        self.target_crs = target_crs
        self.merged_data = None
        self.clusters = None
        self.hotspot_polygons = None
        
    def load_vector_data(self, file_paths, data_dir):
        """
        Load vector data from files and merge into a single GeoDataFrame.

        Returns
        geopandas.GeoDataFrame
            Merged GeoDataFrame of all data loaded from files.

        The function will attempt to convert non-point geometries to points and
        ensure all geometries are in the target CRS. The resulting GeoDataFrame
        will have a 'source' column indicating the file path of the original data.
        """
        gdfs = []
        for file_path in file_paths:
            try:
                full_path = os.path.join(data_dir, file_path) if not os.path.isabs(file_path) else file_path
                gdf = gpd.read_file(full_path)
                
                # Convert to points and ensure CRS
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
    
    def perform_clustering(self, n_clusters=25):
        """
        Perform K-means clustering on the loaded data to group points into n_clusters
        clusters. If n_clusters is greater than the number of points, the number of
        clusters is capped at the number of points.

        Returns:
            geopandas.GeoDataFrame
            A copy of the original data with an additional 'cluster' column indicating
            the cluster assignment of each point.
        """
        if self.merged_data is None or len(self.merged_data) == 0:
            return None
            
        coords = np.array([[point.x, point.y] for point in self.merged_data.geometry])
        actual_clusters = min(n_clusters, len(self.merged_data))
        
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        
        self.merged_data['cluster'] = labels
        self.clusters = self.merged_data.copy()
        
        cluster_counts = pd.Series(labels).value_counts()
        print(f"Created {actual_clusters} clusters from {len(self.merged_data)} points")
        print(f"Points per cluster: min={cluster_counts.min()}, max={cluster_counts.max()}, avg={cluster_counts.mean():.1f}")
        
        return self.clusters
    
    def create_hotspot_polygons(self, buffer_distance=0.3):
        """
        Create polygons for each cluster of points based on the number of points
        in the cluster. If a cluster has 3 or more points, the polygon is the
        convex hull of the points, otherwise it is the unary union of the
        buffers of the points.

        Parameters:
            buffer_distance : float, optional
            Distance to buffer points by when creating polygons. Defaults to 0.3.

        Returns:
            geopandas.GeoDataFrame
            GeoDataFrame of hotspot polygons with columns 'cluster_id' and
            'num_points'.
        """
        if self.clusters is None or len(self.clusters) == 0:
            return None
            
        hotspot_polygons = []
        for cluster_id in self.clusters['cluster'].unique():
            cluster_points = self.clusters[self.clusters['cluster'] == cluster_id]
            points = [Point(row.geometry.x, row.geometry.y) for _, row in cluster_points.iterrows()]
            
            # Create polygon based on point count
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
        print(f"Created {len(self.hotspot_polygons)} hotspot polygons")
        return self.hotspot_polygons
    
    def generate_heatmap_raster(self, output_path, resolution=0.05):
        """
        Generate a heatmap raster of the points in the merged data.

        Parameters:
            output_path : str
                Path to save the heatmap raster to.
            resolution : float, optional
                Resolution of the heatmap raster in the units of the target CRS.
                Defaults to 0.05.

        Returns:
            None
        """
        if self.merged_data is None or len(self.merged_data) == 0:
            return
            
        coords = np.array([[point.x, point.y] for point in self.merged_data.geometry])
        
        # Define bounds with buffer
        buffer = 1.0
        xmin, ymin = coords.min(axis=0) - buffer
        xmax, ymax = coords.max(axis=0) + buffer
        
        # Create grid and calculate KDE
        width, height = int((xmax - xmin) / resolution), int((ymax - ymin) / resolution)
        x_range, y_range = np.linspace(xmin, xmax, width), np.linspace(ymin, ymax, height)
        X, Y = np.meshgrid(x_range, y_range)
        positions = np.vstack([X.ravel(), Y.ravel()])
        
        kde = gaussian_kde(coords.T)
        density = np.flipud(kde(positions).reshape(X.shape))
        
        # Save raster
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
        with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, 
                          count=1, dtype=density.dtype, crs=self.target_crs, transform=transform) as dst:
            dst.write(density, 1)
        
        print(f"Heatmap saved: {output_path}")
    
    def run_analysis(self, file_paths, data_dir, output_dir, n_clusters=25):
        
        """
        Run a comprehensive tourism analysis on the given files.

        This function loads and processes the given files, performs clustering to identify
        tourism hotspots, and generates detailed outputs including hotspot polygons and a
        heatmap raster.

        Args:
            file_paths (list[str]):
                A list of paths to the spatial files to be analyzed.
            data_dir (str):
                The directory containing the spatial files.
            output_dir (str):
                The directory to save the outputs to.
            n_clusters (int, optional):
                The number of clusters to form. Defaults to 25.

        Returns:
            dict:
                A dictionary containing the results of the analysis, including the total
                number of points, clusters created, and hotspot polygons.
        """

        os.makedirs(output_dir, exist_ok=True)
        
        # Load and process data
        self.load_vector_data(file_paths, data_dir)
        if self.merged_data is None or len(self.merged_data) == 0:
            print("No valid data loaded.")
            return None
            
        # Perform clustering and create hotspots
        self.perform_clustering(n_clusters=n_clusters)
        self.create_hotspot_polygons()
        
        # Save outputs
        try:
            self.generate_heatmap_raster(os.path.join(output_dir, 'tourism_heatmap.tiff'))
            
            if self.hotspot_polygons is not None:
                self.hotspot_polygons.to_file(os.path.join(output_dir, 'tourism_hotspots.geojson'))
                print("Saved tourism hotspots")
                
            if self.clusters is not None:
                self.clusters.to_file(os.path.join(output_dir, 'clustered_points.geojson'))
                print("Saved clustered points")
                
        except Exception as e:
            print(f"Error saving outputs: {e}")
        
        print(f"Tourism analysis complete! Output: {output_dir}")
        
        return {
            'total_points': len(self.merged_data),
            'clusters_created': len(self.clusters['cluster'].unique()) if self.clusters is not None else 0,
            'hotspot_polygons': len(self.hotspot_polygons) if self.hotspot_polygons is not None else 0
        }