import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
from osgeo import gdal, osr
import subprocess
import json
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

class TourismHotspotAnalyzer:
    """
    Tourism Hotspot Analysis utility for identifying tourism clusters and generating heat maps
    """
    
    def __init__(self, target_crs='EPSG:4326'):
        """
        Initialize the analyzer
        
        Args:
            target_crs (str): Target coordinate reference system
        """
        self.target_crs = target_crs
        self.merged_data = None
        self.clusters = None
        self.hotspot_polygons = None
        
    def load_vector_data(self, file_paths):
        """
        Load multiple vector datasets and merge them
        
        Args:
            file_paths (list): List of file paths to vector datasets
            
        Returns:
            geopandas.GeoDataFrame: Merged point dataset
        """
        gdfs = []
        
        for file_path in file_paths:
            try:
                print(f"Loading {file_path}...")
                gdf = gpd.read_file(file_path)
                
                # Ensure it's point geometry
                if gdf.geometry.geom_type.iloc[0] not in ['Point', 'MultiPoint']:
                    # Convert to centroid if not points
                    gdf.geometry = gdf.geometry.centroid
                
                # Reproject to target CRS
                if gdf.crs != self.target_crs:
                    gdf = gdf.to_crs(self.target_crs)
                
                # Add source column
                gdf['source'] = os.path.basename(file_path)
                gdfs.append(gdf)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not gdfs:
            raise ValueError("No valid datasets loaded")
        
        # Merge all datasets
        self.merged_data = pd.concat(gdfs, ignore_index=True)
        print(f"Merged {len(self.merged_data)} points from {len(gdfs)} datasets")
        
        return self.merged_data
    
    def perform_clustering(self, eps=0.1, min_samples=3):
        """
        Perform DBSCAN clustering on point data
        
        Args:
            eps (float): Maximum distance between points in a cluster
            min_samples (int): Minimum number of points in a cluster
            
        Returns:
            numpy.array: Cluster labels
        """
        if self.merged_data is None:
            raise ValueError("No data loaded. Call load_vector_data first.")
        
        # Extract coordinates
        coords = np.array([[point.x, point.y] for point in self.merged_data.geometry])
        
        # Perform clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(coords)
        
        # Add cluster labels to data
        self.merged_data['cluster'] = cluster_labels
        
        # Filter out noise points (-1 label)
        self.clusters = self.merged_data[self.merged_data['cluster'] != -1].copy()
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"Found {n_clusters} clusters with {n_noise} noise points")
        
        return cluster_labels
    
    def create_hotspot_polygons(self, buffer_distance=0.05):
        """
        Create polygon representations of hotspot zones
        
        Args:
            buffer_distance (float): Buffer distance around cluster points
            
        Returns:
            geopandas.GeoDataFrame: Hotspot polygons
        """
        if self.clusters is None:
            raise ValueError("No clusters found. Run perform_clustering first.")
        
        hotspot_polygons = []
        
        for cluster_id in self.clusters['cluster'].unique():
            cluster_points = self.clusters[self.clusters['cluster'] == cluster_id]
            
            # Create convex hull of cluster points
            if len(cluster_points) >= 3:
                # For 3+ points, create convex hull
                points = [Point(row.geometry.x, row.geometry.y) for _, row in cluster_points.iterrows()]
                hull = unary_union(points).convex_hull
                
                # Apply buffer
                hotspot_poly = hull.buffer(buffer_distance)
            else:
                # For 1-2 points, create buffer around points
                points = [Point(row.geometry.x, row.geometry.y) for _, row in cluster_points.iterrows()]
                hotspot_poly = unary_union([p.buffer(buffer_distance) for p in points])
            
            hotspot_polygons.append({
                'cluster_id': cluster_id,
                'num_points': len(cluster_points),
                'geometry': hotspot_poly
            })
        
        self.hotspot_polygons = gpd.GeoDataFrame(hotspot_polygons, crs=self.target_crs)
        
        return self.hotspot_polygons
    
    def generate_heatmap_raster(self, output_path, resolution=0.01, extent=None):
        """
        Generate kernel density estimation heatmap as raster
        
        Args:
            output_path (str): Output path for TIFF file
            resolution (float): Pixel resolution in CRS units
            extent (tuple): (xmin, ymin, xmax, ymax) or None for auto
        """
        if self.merged_data is None:
            raise ValueError("No data loaded. Call load_vector_data first.")
        
        # Extract coordinates
        coords = np.array([[point.x, point.y] for point in self.merged_data.geometry])
        
        # Set extent
        if extent is None:
            buffer = 0.1  # 10% buffer
            xmin, ymin = coords.min(axis=0) - buffer
            xmax, ymax = coords.max(axis=0) + buffer
        else:
            xmin, ymin, xmax, ymax = extent
        
        # Create grid
        x_range = np.arange(xmin, xmax, resolution)
        y_range = np.arange(ymin, ymax, resolution)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Flatten for KDE
        positions = np.vstack([X.ravel(), Y.ravel()])
        
        # Perform KDE
        kde = gaussian_kde(coords.T)
        density = kde(positions).reshape(X.shape)
        
        # Create raster using GDAL
        self._save_raster_gdal(density, output_path, xmin, ymax, resolution)
        
        print(f"Heatmap raster saved to: {output_path}")
    
    def _save_raster_gdal(self, array, output_path, x_origin, y_origin, pixel_size):
        """
        Save numpy array as GeoTIFF using GDAL
        """
        rows, cols = array.shape
        
        # Create GeoTIFF
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
        
        # Set geotransform (x_origin, pixel_width, rotation, y_origin, rotation, pixel_height)
        dataset.SetGeoTransform((x_origin, pixel_size, 0, y_origin, 0, -pixel_size))
        
        # Set projection
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # WGS84
        dataset.SetProjection(srs.ExportToWkt())
        
        # Write data
        dataset.GetRasterBand(1).WriteArray(array)
        dataset.GetRasterBand(1).SetNoDataValue(-9999)
        
        # Close dataset
        dataset = None
    
    def save_hotspots_geojson(self, output_path):
        """
        Save hotspot polygons as GeoJSON
        
        Args:
            output_path (str): Output file path
        """
        if self.hotspot_polygons is None:
            raise ValueError("No hotspot polygons created. Run create_hotspot_polygons first.")
        
        self.hotspot_polygons.to_file(output_path, driver='GeoJSON')
        print(f"Hotspot polygons saved to: {output_path}")
    
    def run_complete_analysis(self, file_paths, output_dir, eps=0.1, min_samples=3, 
                             buffer_distance=0.05, resolution=0.01):
        """
        Run complete hotspot analysis pipeline
        
        Args:
            file_paths (list): List of input vector file paths
            output_dir (str): Output directory
            eps (float): DBSCAN eps parameter
            min_samples (int): DBSCAN min_samples parameter
            buffer_distance (float): Buffer distance for hotspot polygons
            resolution (float): Raster resolution for heatmap
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Load and merge data
        print("Step 1: Loading and merging datasets...")
        self.load_vector_data(file_paths)
        
        # Step 2: Perform clustering
        print("Step 2: Performing clustering...")
        self.perform_clustering(eps=eps, min_samples=min_samples)
        
        # Step 3: Create hotspot polygons
        print("Step 3: Creating hotspot polygons...")
        self.create_hotspot_polygons(buffer_distance=buffer_distance)
        
        # Step 4: Generate heatmap
        print("Step 4: Generating heatmap raster...")
        heatmap_path = os.path.join(output_dir, 'tourism_heatmap.tiff')
        self.generate_heatmap_raster(heatmap_path, resolution=resolution)
        
        # Step 5: Save outputs
        print("Step 5: Saving outputs...")
        hotspots_path = os.path.join(output_dir, 'tourism_hotspots.geojson')
        self.save_hotspots_geojson(hotspots_path)
        
        # Save clustered points
        clusters_path = os.path.join(output_dir, 'clustered_points.geojson')
        if self.clusters is not None:
            self.clusters.to_file(clusters_path, driver='GeoJSON')
        
        print(f"\nAnalysis complete! Outputs saved to: {output_dir}")
        return {
            'hotspots': hotspots_path,
            'heatmap': heatmap_path,
            'clusters': clusters_path
        }

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TourismHotspotAnalyzer()
    
    # Example file paths (update with actual paths)
    file_paths = [
        "path/to/wildlife_sanctuaries.gpkg",
        "path/to/mountain_passes.geojson",
        "path/to/zoos.gpkg"
    ]
    
    # Run analysis
    try:
        results = analyzer.run_complete_analysis(
            file_paths=file_paths,
            output_dir="./tourism_analysis_output",
            eps=0.1,  # Adjust based on your data scale
            min_samples=3,
            buffer_distance=0.05,
            resolution=0.01
        )
        print("Analysis completed successfully!")
        print("Results:", results)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
