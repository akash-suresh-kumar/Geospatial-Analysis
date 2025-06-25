import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import os
import warnings
warnings.filterwarnings('ignore')

class ServiceGapAnalyzer:
    """
    Service Gap Analysis utility for identifying well-served and underserved areas
    """
    
    def __init__(self, target_crs='EPSG:4326'):
        """
        Initialize the analyzer
        
        Args:
            target_crs (str): Target coordinate reference system
        """
        self.target_crs = target_crs
        self.pois = None
        self.village_boundaries = None
        self.poi_clusters = None
        self.served_areas = None
        self.underserved_areas = None
        
    def load_data(self, poi_path, village_boundary_path):
        """
        Load POI and village boundary data
        
        Args:
            poi_path (str): Path to POI dataset
            village_boundary_path (str): Path to village boundary dataset
        """
        print("Loading POI data...")
        self.pois = gpd.read_file(poi_path)
        
        print("Loading village boundaries...")
        self.village_boundaries = gpd.read_file(village_boundary_path)
        
        # Check and standardize CRS
        if self.pois.crs != self.target_crs:
            print(f"Reprojecting POIs from {self.pois.crs} to {self.target_crs}")
            self.pois = self.pois.to_crs(self.target_crs)
            
        if self.village_boundaries.crs != self.target_crs:
            print(f"Reprojecting village boundaries from {self.village_boundaries.crs} to {self.target_crs}")
            self.village_boundaries = self.village_boundaries.to_crs(self.target_crs)
        
        print(f"Loaded {len(self.pois)} POIs and {len(self.village_boundaries)} village boundaries")
        
        # Ensure POIs are points
        if not all(self.pois.geometry.geom_type == 'Point'):
            print("Converting non-point geometries to centroids...")
            self.pois.geometry = self.pois.geometry.centroid
    
    def cluster_pois(self, algorithm='DBSCAN', eps=0.01, min_samples=3, min_cluster_size=5):
        """
        Cluster POIs using DBSCAN or HDBSCAN
        
        Args:
            algorithm (str): 'DBSCAN' or 'HDBSCAN'
            eps (float): DBSCAN eps parameter
            min_samples (int): Minimum samples parameter
            min_cluster_size (int): HDBSCAN minimum cluster size
            
        Returns:
            numpy.array: Cluster labels
        """
        if self.pois is None:
            raise ValueError("POIs not loaded. Call load_data first.")
        
        # Extract coordinates
        coords = np.array([[point.x, point.y] for point in self.pois.geometry])
        
        # Choose clustering algorithm
        if algorithm.upper() == 'DBSCAN':
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        elif algorithm.upper() == 'HDBSCAN':
            clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        else:
            raise ValueError("Algorithm must be 'DBSCAN' or 'HDBSCAN'")
        
        # Perform clustering
        cluster_labels = clusterer.fit_predict(coords)
        
        # Add cluster labels to POIs
        self.pois['cluster'] = cluster_labels
        
        # Filter out noise points (-1 label)
        self.poi_clusters = self.pois[self.pois['cluster'] != -1].copy()
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"Found {n_clusters} POI clusters with {n_noise} noise points")
        
        return cluster_labels
    
    def create_service_areas(self, method='convex_hull', buffer_distance=0.02):
        """
        Create service area polygons from POI clusters
        
        Args:
            method (str): 'convex_hull', 'buffer_union', or 'both'
            buffer_distance (float): Buffer distance around service areas
            
        Returns:
            geopandas.GeoDataFrame: Service area polygons
        """
        if self.poi_clusters is None:
            raise ValueError("POI clusters not created. Run cluster_pois first.")
        
        service_areas = []
        
        for cluster_id in self.poi_clusters['cluster'].unique():
            cluster_pois = self.poi_clusters[self.poi_clusters['cluster'] == cluster_id]
            
            # Get cluster points
            points = [Point(row.geometry.x, row.geometry.y) for _, row in cluster_pois.iterrows()]
            
            if len(points) == 1:
                # Single point - create buffer
                service_area = points[0].buffer(buffer_distance)
            elif len(points) == 2:
                # Two points - create buffer union
                service_area = unary_union([p.buffer(buffer_distance) for p in points])
            else:
                # Multiple points
                if method == 'convex_hull':
                    hull = unary_union(points).convex_hull
                    service_area = hull.buffer(buffer_distance)
                elif method == 'buffer_union':
                    service_area = unary_union([p.buffer(buffer_distance) for p in points])
                else:  # both
                    hull = unary_union(points).convex_hull
                    buffered_points = unary_union([p.buffer(buffer_distance) for p in points])
                    service_area = unary_union([hull, buffered_points])
            
            service_areas.append({
                'cluster_id': cluster_id,
                'num_pois': len(cluster_pois),
                'geometry': service_area
            })
        
        service_areas_gdf = gpd.GeoDataFrame(service_areas, crs=self.target_crs)
        
        # Merge overlapping service areas
        print("Merging overlapping service areas...")
        self.served_areas = gpd.GeoDataFrame({
            'service_type': ['merged_service_areas'],
            'geometry': [unary_union(service_areas_gdf.geometry)]
        }, crs=self.target_crs)
        
        print(f"Created {len(service_areas_gdf)} individual service areas, merged into {len(self.served_areas)} unified areas")
        
        return service_areas_gdf
    
    def identify_underserved_areas(self):
        """
        Identify underserved areas by subtracting served areas from village boundaries
        
        Returns:
            geopandas.GeoDataFrame: Underserved area polygons
        """
        if self.served_areas is None:
            raise ValueError("Service areas not created. Run create_service_areas first.")
        
        if self.village_boundaries is None:
            raise ValueError("Village boundaries not loaded. Call load_data first.")
        
        print("Identifying underserved areas...")
        
        # Create a unified village boundary
        unified_villages = unary_union(self.village_boundaries.geometry)
        unified_served = unary_union(self.served_areas.geometry)
        
        # Calculate spatial difference
        underserved_geometry = unified_villages.difference(unified_served)
        
        # Handle MultiPolygon results
        if hasattr(underserved_geometry, 'geoms'):
            # MultiPolygon - create separate records for each polygon
            underserved_polys = []
            for i, geom in enumerate(underserved_geometry.geoms):
                if geom.area > 0:  # Filter out tiny polygons
                    underserved_polys.append({
                        'underserved_id': i,
                        'area': geom.area,
                        'geometry': geom
                    })
        else:
            # Single Polygon
            underserved_polys = [{
                'underserved_id': 0,
                'area': underserved_geometry.area,
                'geometry': underserved_geometry
            }]
        
        self.underserved_areas = gpd.GeoDataFrame(underserved_polys, crs=self.target_crs)
        
        # Filter out very small areas (potential artifacts)
        min_area_threshold = self.underserved_areas['area'].quantile(0.1)  # Bottom 10%
        self.underserved_areas = self.underserved_areas[
            self.underserved_areas['area'] > min_area_threshold
        ].reset_index(drop=True)
        
        print(f"Identified {len(self.underserved_areas)} underserved areas")
        
        return self.underserved_areas
    
    def calculate_coverage_statistics(self):
        """
        Calculate coverage statistics
        
        Returns:
            dict: Coverage statistics
        """
        if self.served_areas is None or self.underserved_areas is None:
            raise ValueError("Analysis not complete. Run full analysis first.")
        
        total_area = unary_union(self.village_boundaries.geometry).area
        served_area = unary_union(self.served_areas.geometry).area
        underserved_area = self.underserved_areas['area'].sum()
        
        coverage_stats = {
            'total_village_area': total_area,
            'served_area': served_area,
            'underserved_area': underserved_area,
            'coverage_percentage': (served_area / total_area) * 100,
            'underserved_percentage': (underserved_area / total_area) * 100,
            'num_poi_clusters': len(self.poi_clusters['cluster'].unique()),
            'num_underserved_areas': len(self.underserved_areas)
        }
        
        return coverage_stats
    
    def save_outputs(self, output_dir):
        """
        Save analysis outputs as GeoJSON files
        
        Args:
            output_dir (str): Output directory path
            
        Returns:
            dict: Output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = {}
        
        # Save served areas
        if self.served_areas is not None:
            served_path = os.path.join(output_dir, 'well_served_areas.geojson')
            self.served_areas.to_file(served_path, driver='GeoJSON')
            output_paths['served_areas'] = served_path
            print(f"Well-served areas saved to: {served_path}")
        
        # Save underserved areas
        if self.underserved_areas is not None:
            underserved_path = os.path.join(output_dir, 'underserved_areas.geojson')
            self.underserved_areas.to_file(underserved_path, driver='GeoJSON')
            output_paths['underserved_areas'] = underserved_path
            print(f"Underserved areas saved to: {underserved_path}")
        
        # Save POI clusters
        if self.poi_clusters is not None:
            clusters_path = os.path.join(output_dir, 'poi_clusters.geojson')
            self.poi_clusters.to_file(clusters_path, driver='GeoJSON')
            output_paths['poi_clusters'] = clusters_path
            print(f"POI clusters saved to: {clusters_path}")
        
        return output_paths
    
    def run_complete_analysis(self, poi_path, village_boundary_path, output_dir,
                             algorithm='DBSCAN', eps=0.01, min_samples=3,
                             service_area_method='convex_hull', buffer_distance=0.02):
        """
        Run complete service gap analysis pipeline
        
        Args:
            poi_path (str): Path to POI dataset
            village_boundary_path (str): Path to village boundary dataset
            output_dir (str): Output directory
            algorithm (str): Clustering algorithm ('DBSCAN' or 'HDBSCAN')
            eps (float): DBSCAN eps parameter
            min_samples (int): Minimum samples parameter
            service_area_method (str): Service area creation method
            buffer_distance (float): Buffer distance for service areas
            
        Returns:
            dict: Analysis results and output paths
        """
        print("=== Starting Service Gap Analysis ===")
        
        # Step 1: Load data
        print("\nStep 1: Loading data...")
        self.load_data(poi_path, village_boundary_path)
        
        # Step 2: Cluster POIs
        print("\nStep 2: Clustering POIs...")
        self.cluster_pois(algorithm=algorithm, eps=eps, min_samples=min_samples)
        
        # Step 3: Create service areas
        print("\nStep 3: Creating service areas...")
        self.create_service_areas(method=service_area_method, buffer_distance=buffer_distance)
        
        # Step 4: Identify underserved areas
        print("\nStep 4: Identifying underserved areas...")
        self.identify_underserved_areas()
        
        # Step 5: Calculate statistics
        print("\nStep 5: Calculating coverage statistics...")
        stats = self.calculate_coverage_statistics()
        
        # Step 6: Save outputs
        print("\nStep 6: Saving outputs...")
        output_paths = self.save_outputs(output_dir)
        
        # Print summary
        print("\n=== Analysis Summary ===")
        print(f"Total Village Area: {stats['total_village_area']:.2f} sq units")
        print(f"Coverage Percentage: {stats['coverage_percentage']:.1f}%")
        print(f"Underserved Percentage: {stats['underserved_percentage']:.1f}%")
        print(f"Number of POI Clusters: {stats['num_poi_clusters']}")
        print(f"Number of Underserved Areas: {stats['num_underserved_areas']}")
        
        return {
            'statistics': stats,
            'output_paths': output_paths
        }

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ServiceGapAnalyzer()
    
    # Example file paths (update with actual paths)
    poi_path = "path/to/points_of_interest.gpkg"
    village_boundary_path = "path/to/village_boundaries.gpkg"
    
    # Run analysis
    try:
        results = analyzer.run_complete_analysis(
            poi_path=poi_path,
            village_boundary_path=village_boundary_path,
            output_dir="./service_gap_analysis_output",
            algorithm='DBSCAN',  # or 'HDBSCAN'
            eps=0.01,  # Adjust based on your data scale
            min_samples=3,
            service_area_method='convex_hull',  # or 'buffer_union' or 'both'
            buffer_distance=0.02
        )
        
        print("\nAnalysis completed successfully!")
        print("Results:", results)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
