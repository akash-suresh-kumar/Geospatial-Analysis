import os, sys, glob, warnings
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point
from shapely.ops import unary_union
warnings.filterwarnings('ignore')

try:
    from shapely import make_valid
except ImportError:
    make_valid = lambda geom: geom.buffer(0) if geom and not geom.is_empty else geom

def get_data_dir(): return os.path.join(os.getcwd(), 'data')
def get_output_dir(): return os.path.join(os.getcwd(), 'output')

def find_files_by_keywords(data_dir, keywords):
    files = []
    for ext in ['*.shp', '*.geojson', '*.json', '*.gpkg']:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    return [os.path.basename(f) for f in files if any(k.lower() in os.path.basename(f).lower() for k in keywords)]

def list_spatial_files(data_dir):
    files = []
    for ext in ['*.shp', '*.geojson', '*.json', '*.gpkg']:
        files.extend([os.path.basename(f) for f in glob.glob(os.path.join(data_dir, ext))])
    return files

class TourismHotspotAnalyzer:
    def __init__(self, target_crs='EPSG:4326'):
        self.target_crs = target_crs
        self.merged_data = self.clusters = self.hotspot_polygons = None
        
    def load_vector_data(self, file_paths, data_dir):
        gdfs = []
        for file_path in file_paths:
            try:
                full_path = os.path.join(data_dir, file_path) if not os.path.isabs(file_path) else file_path
                gdf = gpd.read_file(full_path)
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
        if not self.merged_data is None and len(self.merged_data) > 0:
            coords = np.array([[p.x, p.y] for p in self.merged_data.geometry])
            actual_clusters = min(n_clusters, len(self.merged_data))
            labels = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10).fit_predict(coords)
            self.merged_data['cluster'] = labels
            self.clusters = self.merged_data.copy()
            cluster_counts = pd.Series(labels).value_counts()
            print(f"Created {actual_clusters} clusters from {len(self.merged_data)} points")
            print(f"Points per cluster: min={cluster_counts.min()}, max={cluster_counts.max()}, avg={cluster_counts.mean():.1f}")
        return self.clusters
    
    def create_hotspot_polygons(self, buffer_distance=0.3):
        if not self.clusters is None and len(self.clusters) > 0:
            hotspot_polygons = []
            for cluster_id in self.clusters['cluster'].unique():
                cluster_points = self.clusters[self.clusters['cluster'] == cluster_id]
                points = [Point(row.geometry.x, row.geometry.y) for _, row in cluster_points.iterrows()]
                hull = unary_union(points).convex_hull.buffer(buffer_distance) if len(points) >= 3 else unary_union([p.buffer(buffer_distance) for p in points])
                hotspot_polygons.append({'cluster_id': cluster_id, 'num_points': len(cluster_points), 'geometry': hull})
            self.hotspot_polygons = gpd.GeoDataFrame(hotspot_polygons, crs=self.target_crs)
            print(f"Created {len(self.hotspot_polygons)} hotspot polygons")
        return self.hotspot_polygons
    
    def generate_heatmap_raster(self, output_path, resolution=0.05):
        if not self.merged_data is None and len(self.merged_data) > 0:
            coords = np.array([[p.x, p.y] for p in self.merged_data.geometry])
            buffer = 1.0
            xmin, ymin = coords.min(axis=0) - buffer
            xmax, ymax = coords.max(axis=0) + buffer
            width, height = int((xmax - xmin) / resolution), int((ymax - ymin) / resolution)
            x_range, y_range = np.linspace(xmin, xmax, width), np.linspace(ymin, ymax, height)
            X, Y = np.meshgrid(x_range, y_range)
            positions = np.vstack([X.ravel(), Y.ravel()])
            density = np.flipud(gaussian_kde(coords.T)(positions).reshape(X.shape))
            transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
            with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=density.dtype, crs=self.target_crs, transform=transform) as dst:
                dst.write(density, 1)
            print(f"Heatmap saved: {output_path}")
    
    def run_analysis(self, file_paths, data_dir, output_dir, n_clusters=25, create_heatmap=True, resolution=0.05):
        os.makedirs(output_dir, exist_ok=True)
        self.load_vector_data(file_paths, data_dir)
        if self.merged_data is None or len(self.merged_data) == 0:
            print("No valid data loaded.")
            return None
        
        self.perform_clustering(n_clusters=n_clusters)
        self.create_hotspot_polygons()
        
        try:
            if create_heatmap:
                self.generate_heatmap_raster(os.path.join(output_dir, 'tourism_heatmap.tiff'), resolution)
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

class ServiceGapAnalyzer:
    def __init__(self, target_crs='EPSG:4326'):
        self.target_crs = target_crs
        self.pois = self.boundaries = self.served_areas = self.underserved_areas = self.projected_crs = None

    def _get_projected_crs(self, gdf):
        if gdf.crs is None or gdf.empty:
            return self.target_crs
        centroid = gdf.geometry.centroid.iloc[0]
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = 'north' if centroid.y >= 0 else 'south'
        return f'EPSG:326{utm_zone}' if hemisphere == 'north' else f'EPSG:327{utm_zone}'

    def load_data(self, poi_path, boundary_path, data_dir):
        try:
            poi_full_path = os.path.join(data_dir, poi_path) if not os.path.isabs(poi_path) else poi_path
            self.pois = gpd.read_file(poi_full_path)
            if self.pois.crs != self.target_crs:
                self.pois = self.pois.to_crs(self.target_crs)
            if not all(self.pois.geometry.geom_type == 'Point'):
                self.pois.geometry = self.pois.geometry.centroid
            print(f"Loaded {len(self.pois)} POIs")
        except Exception as e:
            print(f"Error loading POI file {poi_path}: {e}")
            return False
        
        try:
            boundary_full_path = os.path.join(data_dir, boundary_path) if not os.path.isabs(boundary_path) else boundary_path
            self.boundaries = gpd.read_file(boundary_full_path)
            if self.boundaries.crs != self.target_crs:
                self.boundaries = self.boundaries.to_crs(self.target_crs)
            self.projected_crs = self._get_projected_crs(self.boundaries)
            print(f"Using projected CRS: {self.projected_crs}")
            print(f"Loaded {len(self.boundaries)} boundaries")
        except Exception as e:
            print(f"Error loading boundary file {boundary_path}: {e}")
            return False
        return True
    
    def find_optimal_eps(self, target_clusters=45, k=4):
        if self.pois is None or len(self.pois) == 0:
            return 0.01
        coords = np.array([[p.x, p.y] for p in self.pois.geometry])
        neighbors = NearestNeighbors(n_neighbors=min(k, len(coords)))
        distances, _ = neighbors.fit(coords).kneighbors(coords)
        eps_candidates = np.percentile(np.sort(distances[:, -1]), [50, 60, 70, 80, 85, 90, 95])
        best_eps, best_cluster_count = 0.01, 0
        for eps in eps_candidates:
            labels = DBSCAN(eps=eps, min_samples=2).fit_predict(coords)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if abs(n_clusters - target_clusters) < abs(best_cluster_count - target_clusters):
                best_eps, best_cluster_count = eps, n_clusters
        print(f"Selected eps={best_eps:.4f} for ~{best_cluster_count} clusters")
        return best_eps
    
    def cluster_pois(self, eps=None, min_samples=2, target_clusters=45):
        if self.pois is None or len(self.pois) == 0:
            print("No POIs available for clustering")
            return None
        
        if eps is None:
            eps = self.find_optimal_eps(target_clusters=target_clusters)
        coords = np.array([[p.x, p.y] for p in self.pois.geometry])
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
        self.pois['cluster'] = labels
        poi_clusters = self.pois[self.pois['cluster'] != -1].copy()
        outliers = self.pois[self.pois['cluster'] == -1].copy()
        
        if len(outliers) > 0:
            max_cluster = poi_clusters['cluster'].max() if len(poi_clusters) > 0 else -1
            outliers['cluster'] = range(max_cluster + 1, max_cluster + 1 + len(outliers))
            poi_clusters = gpd.GeoDataFrame(pd.concat([poi_clusters, outliers], ignore_index=True), crs=self.target_crs)
        
        n_real_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = sum(1 for x in labels if x == -1)
        print(f"Clustering results: {n_real_clusters} clusters, {n_outliers} individual POIs")
        return poi_clusters
    
    def create_service_areas(self, poi_clusters, buffer_distance=0.3):
        if poi_clusters is None or len(poi_clusters) == 0:
            return None
        
        service_areas = []
        for cluster_id in poi_clusters['cluster'].unique():
            cluster_pois = poi_clusters[poi_clusters['cluster'] == cluster_id]
            points = [Point(row.geometry.x, row.geometry.y) for _, row in cluster_pois.iterrows()]
            service_area = make_valid(unary_union(points).convex_hull.buffer(buffer_distance) if len(points) >= 3 else unary_union([p.buffer(buffer_distance) for p in points]))
            service_areas.append({'cluster_id': cluster_id, 'geometry': service_area, 'poi_count': len(points)})
        
        service_areas_gdf = gpd.GeoDataFrame(service_areas, crs=self.target_crs)
        valid_geoms = [geom for geom in service_areas_gdf.geometry if geom and not geom.is_empty and geom.is_valid]
        
        if valid_geoms:
            unified_service_area = make_valid(unary_union(valid_geoms))
            self.served_areas = gpd.GeoDataFrame({'geometry': [unified_service_area]}, crs=self.target_crs)
        else:
            self.served_areas = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=self.target_crs), crs=self.target_crs)
        
        print(f"Created {len(service_areas_gdf)} service areas")
        return service_areas_gdf
    
    def identify_underserved_areas(self, min_area_threshold=1e-8):
        if self.boundaries is None or self.served_areas is None:
            return None
        
        boundaries_proj = self.boundaries.to_crs(self.projected_crs)
        served_areas_proj = self.served_areas.to_crs(self.projected_crs)
        unified_boundaries = make_valid(unary_union(boundaries_proj.geometry))
        
        if len(served_areas_proj) > 0 and not served_areas_proj.geometry.iloc[0].is_empty:
            unified_served = make_valid(unary_union(served_areas_proj.geometry))
            underserved_geometry = unified_boundaries.difference(unified_served)
        else:
            underserved_geometry = unified_boundaries
        
        underserved_polys = []
        geoms = underserved_geometry.geoms if hasattr(underserved_geometry, 'geoms') else [underserved_geometry]
        
        for geom in geoms:
            if (geom.is_valid and hasattr(geom, 'area') and geom.area > min_area_threshold and geom.geom_type in ['Polygon', 'MultiPolygon']):
                underserved_polys.append({'geometry': geom})
        
        if underserved_polys:
            underserved_gdf = gpd.GeoDataFrame(underserved_polys, crs=self.projected_crs)
            self.underserved_areas = underserved_gdf.to_crs(self.target_crs)
        else:
            self.underserved_areas = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=self.target_crs), crs=self.target_crs)
        
        print(f"Identified {len(self.underserved_areas)} underserved areas")
        return self.underserved_areas
    
    def calculate_coverage_stats(self):
        if self.boundaries is None:
            return None
        
        boundaries_projected = self.boundaries.to_crs(self.projected_crs)
        total_area = make_valid(unary_union(boundaries_projected.geometry)).area
        served_area = 0
        
        if self.served_areas is not None and len(self.served_areas) > 0:
            served_areas_projected = self.served_areas.to_crs(self.projected_crs)
            if not served_areas_projected.geometry.iloc[0].is_empty:
                served_area = make_valid(unary_union(served_areas_projected.geometry)).area
        
        coverage_pct = min((served_area / total_area * 100) if total_area > 0 else 0, 100.0)
        return {
            'total_area': total_area,
            'served_area': served_area,
            'coverage_percentage': coverage_pct,
            'underserved_areas_count': len(self.underserved_areas) if self.underserved_areas is not None else 0
        }
    
    def run_analysis(self, poi_path, boundary_path, data_dir, output_dir, eps=None, buffer_distance=0.3, target_clusters=45):
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.load_data(poi_path, boundary_path, data_dir):
            return None
        
        poi_clusters = self.cluster_pois(eps=eps, target_clusters=target_clusters)
        if poi_clusters is None:
            return None
        
        service_areas = self.create_service_areas(poi_clusters, buffer_distance)
        if service_areas is None:
            return None
        
        self.identify_underserved_areas()
        stats = self.calculate_coverage_stats()
        
        try:
            if self.served_areas is not None and len(self.served_areas) > 0:
                self.served_areas.to_file(os.path.join(output_dir, 'well_served_areas.geojson'))
            if self.underserved_areas is not None and len(self.underserved_areas) > 0:
                self.underserved_areas.to_file(os.path.join(output_dir, 'underserved_areas.geojson'))
            if poi_clusters is not None and len(poi_clusters) > 0:
                poi_clusters.to_file(os.path.join(output_dir, 'poi_clusters.geojson'))
            if service_areas is not None and len(service_areas) > 0:
                service_areas.to_file(os.path.join(output_dir, 'service_areas.geojson'))
        except Exception as e:
            print(f"Error saving outputs: {e}")
        
        if stats:
            print(f"\nSERVICE GAP ANALYSIS COMPLETE!")
            print(f"POIs processed: {len(self.pois)}")
            print(f"Coverage: {stats['coverage_percentage']:.1f}%")
            print(f"Underserved areas: {stats['underserved_areas_count']}")
        
        return stats