import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
from shapely.ops import unary_union
import os
import warnings
warnings.filterwarnings('ignore')

class ServiceGapAnalyzer:
    def __init__(self, target_crs='EPSG:4326'):
        self.target_crs = target_crs
        self.pois = None
        self.boundaries = None
        self.served_areas = None
        self.underserved_areas = None
        
    def load_data(self, poi_path, boundary_path, data_dir):
        try:
            # Load POI data
            poi_full_path = os.path.join(data_dir, poi_path) if not os.path.isabs(poi_path) else poi_path
            self.pois = gpd.read_file(poi_full_path)
            
            if self.pois.crs != self.target_crs:
                self.pois = self.pois.to_crs(self.target_crs)
            
            # Convert to points if needed
            if not all(self.pois.geometry.geom_type == 'Point'):
                self.pois.geometry = self.pois.geometry.centroid
            
            print(f"Loaded {len(self.pois)} POIs")
        except Exception as e:
            print(f"Error loading POI file: {e}")
            return False
        
        try:
            # Load boundary data
            boundary_full_path = os.path.join(data_dir, boundary_path) if not os.path.isabs(boundary_path) else boundary_path
            self.boundaries = gpd.read_file(boundary_full_path)
            
            if self.boundaries.crs != self.target_crs:
                self.boundaries = self.boundaries.to_crs(self.target_crs)
            
            print(f"Loaded {len(self.boundaries)} boundaries")
        except Exception as e:
            print(f"Error loading boundary file: {e}")
            return False
            
        return True
    
    def cluster_pois(self, eps=0.01, min_samples=2):
        if self.pois is None or len(self.pois) == 0:
            return None
            
        coords = np.array([[point.x, point.y] for point in self.pois.geometry])
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(coords)
        
        self.pois['cluster'] = cluster_labels
        poi_clusters = self.pois[self.pois['cluster'] != -1].copy()
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"Found {n_clusters} POI clusters")
        return poi_clusters
    
    def create_service_areas(self, poi_clusters, buffer_distance=0.02):
        if poi_clusters is None or len(poi_clusters) == 0:
            return None
            
        service_areas = []
        
        for cluster_id in poi_clusters['cluster'].unique():
            cluster_pois = poi_clusters[poi_clusters['cluster'] == cluster_id]
            points = [Point(row.geometry.x, row.geometry.y) for _, row in cluster_pois.iterrows()]
            
            if len(points) >= 3:
                service_area = unary_union(points).convex_hull.buffer(buffer_distance)
            else:
                service_area = unary_union([p.buffer(buffer_distance) for p in points])
            
            service_areas.append({'cluster_id': cluster_id, 'geometry': service_area})
        
        service_areas_gdf = gpd.GeoDataFrame(service_areas, crs=self.target_crs)
        self.served_areas = gpd.GeoDataFrame({
            'geometry': [unary_union(service_areas_gdf.geometry)]
        }, crs=self.target_crs)
        
        return service_areas_gdf
    
    def identify_underserved_areas(self):
        if self.boundaries is None or self.served_areas is None:
            return None
            
        unified_boundaries = unary_union(self.boundaries.geometry)
        unified_served = unary_union(self.served_areas.geometry)
        underserved_geometry = unified_boundaries.difference(unified_served)
        
        if hasattr(underserved_geometry, 'geoms'):
            underserved_polys = [{'geometry': geom} for geom in underserved_geometry.geoms 
                               if geom.area > 0]
        else:
            underserved_polys = [{'geometry': underserved_geometry}] if underserved_geometry.area > 0 else []
        
        self.underserved_areas = gpd.GeoDataFrame(underserved_polys, crs=self.target_crs)
        print(f"Identified {len(self.underserved_areas)} underserved areas")
        return self.underserved_areas
    
    def calculate_coverage_stats(self):
        if self.boundaries is None or self.served_areas is None:
            return None
            
        total_area = unary_union(self.boundaries.geometry).area
        served_area = unary_union(self.served_areas.geometry).area
        coverage_pct = (served_area / total_area) * 100
        
        return {
            'total_area': total_area,
            'served_area': served_area,
            'coverage_percentage': coverage_pct,
            'underserved_areas_count': len(self.underserved_areas) if self.underserved_areas is not None else 0
        }
    
    def run_analysis(self, poi_path, boundary_path, data_dir, output_dir, eps=0.01, buffer_distance=0.02):
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.load_data(poi_path, boundary_path, data_dir):
            return None
        
        poi_clusters = self.cluster_pois(eps=eps)
        if poi_clusters is None:
            return None
            
        service_areas = self.create_service_areas(poi_clusters, buffer_distance)
        if service_areas is None:
            return None
        
        self.identify_underserved_areas()
        stats = self.calculate_coverage_stats()
        
        # Save outputs
        try:
            if self.served_areas is not None and len(self.served_areas) > 0:
                self.served_areas.to_file(os.path.join(output_dir, 'well_served_areas.geojson'))
        except Exception as e:
            print(f"Error saving served areas: {e}")
            
        try:
            if self.underserved_areas is not None and len(self.underserved_areas) > 0:
                self.underserved_areas.to_file(os.path.join(output_dir, 'underserved_areas.geojson'))
        except Exception as e:
            print(f"Error saving underserved areas: {e}")
            
        try:
            if poi_clusters is not None and len(poi_clusters) > 0:
                poi_clusters.to_file(os.path.join(output_dir, 'poi_clusters.geojson'))
        except Exception as e:
            print(f"Error saving POI clusters: {e}")
        
        if stats:
            print(f"Service Gap Analysis Complete!")
            print(f"Coverage: {stats['coverage_percentage']:.1f}%")
            print(f"Underserved areas: {stats['underserved_areas_count']}")
            print(f"Output: {output_dir}")
        
        return stats