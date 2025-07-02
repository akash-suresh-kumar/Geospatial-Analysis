import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Point
from shapely.ops import unary_union
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from shapely import make_valid
except ImportError:
    def make_valid(geom):
        return geom.buffer(0) if geom and not geom.is_empty else geom

class ServiceGapAnalyzer:
    def __init__(self, target_crs='EPSG:4326'):
        """
        Initialize ServiceGapAnalyzer.

        Parameters
        ----------
        target_crs : str, optional
            Target coordinate reference system for the analysis. Defaults to 'EPSG:4326'.

        Returns
        -------
        ServiceGapAnalyzer
            An instance of ServiceGapAnalyzer.
        """
        self.target_crs = target_crs
        self.pois = None
        self.boundaries = None
        self.served_areas = None
        self.underserved_areas = None
        self.projected_crs = None

    def _get_projected_crs(self, gdf):
        """
        Get a projected CRS appropriate for the given geodataframe.

        Parameters
        ----------
        gdf : GeoDataFrame

        Returns
        -------
        projected_crs : str
            A projected CRS appropriate for the given geodataframe.
            Returns the target_crs if the geodataframe is empty or has no CRS.
        """
        if gdf.crs is None or gdf.empty:
            return self.target_crs
        centroid = gdf.geometry.centroid.iloc[0]
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = 'north' if centroid.y >= 0 else 'south'
        return f'EPSG:326{utm_zone}' if hemisphere == 'north' else f'EPSG:327{utm_zone}'

    def load_data(self, poi_path, boundary_path, data_dir):
        
        """
        Load POI and boundary data from files.

        Parameters
        ----------
        poi_path : str
            Path to POI file. If not absolute, data_dir is prepended.
        boundary_path : str
            Path to boundary file. If not absolute, data_dir is prepended.
        data_dir : str
            Directory path to prepend to poi_path and boundary_path if not absolute.

        Returns
        -------
        bool
            True if all files loaded successfully, False otherwise.
        """
        
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
            print(f"Error loading POI file {poi_path}: {e}")
            return False
        
        try:
            # Load boundary data
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
        """
        Find the optimal epsilon value for DBSCAN clustering that results in the
        closest number of clusters to the target number.

        Parameters
        ----------
        target_clusters : int
            The target number of clusters.
        k : int
            The number of nearest neighbors to consider when calculating the
            density of each point.

        Returns
        -------
        float
            The optimal epsilon value.
        """
        if self.pois is None or len(self.pois) == 0:
            return 0.01
            
        coords = np.array([[point.x, point.y] for point in self.pois.geometry])
        
        # Calculate k-nearest neighbors distances
        neighbors = NearestNeighbors(n_neighbors=min(k, len(coords)))
        distances, _ = neighbors.fit(coords).kneighbors(coords)
        
        # Try different eps values
        eps_candidates = np.percentile(np.sort(distances[:, -1]), [50, 60, 70, 80, 85, 90, 95])
        
        best_eps, best_cluster_count = 0.01, 0
        
        for eps in eps_candidates:
            labels = DBSCAN(eps=eps, min_samples=2).fit_predict(coords)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            print(f"eps={eps:.4f} -> {n_clusters} clusters")
            
            if abs(n_clusters - target_clusters) < abs(best_cluster_count - target_clusters):
                best_eps, best_cluster_count = eps, n_clusters
        
        print(f"Selected eps={best_eps:.4f} for ~{best_cluster_count} clusters")
        return best_eps
    
    def cluster_pois(self, eps=None, min_samples=2, target_clusters=45):
        
        """
        Perform DBSCAN clustering on the loaded POIs with the given parameters.

        Parameters:
            eps (float): maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            target_clusters (int): target number of clusters to be formed

        Returns:
            geopandas.GeoDataFrame: a copy of the original POI data with an additional 'cluster' column indicating the cluster assignment of each point
        """

        if self.pois is None or len(self.pois) == 0:
            print("No POIs available for clustering")
            return None
        
        if eps is None:
            eps = self.find_optimal_eps(target_clusters=target_clusters)
            
        coords = np.array([[point.x, point.y] for point in self.pois.geometry])
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
        
        self.pois['cluster'] = labels
        
        # Handle clustered POIs and outliers
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
        
        """
        Create service areas by buffering POIs in each cluster.
        
        Args:
            poi_clusters (GeoDataFrame): Clusters of POIs.
            buffer_distance (float, optional): Distance to buffer each POI. Defaults to 0.3.
        
        Returns:
            GeoDataFrame: A GeoDataFrame containing the service areas for each cluster.
        """
        if poi_clusters is None or len(poi_clusters) == 0:
            return None
            
        service_areas = []
        
        for cluster_id in poi_clusters['cluster'].unique():
            cluster_pois = poi_clusters[poi_clusters['cluster'] == cluster_id]
            points = [Point(row.geometry.x, row.geometry.y) for _, row in cluster_pois.iterrows()]
            
            if len(points) >= 3:
                service_area = make_valid(unary_union(points).convex_hull.buffer(buffer_distance))
            else:
                service_area = make_valid(unary_union([p.buffer(buffer_distance) for p in points]))
            
            service_areas.append({
                'cluster_id': cluster_id, 
                'geometry': service_area,
                'poi_count': len(points)
            })
        
        service_areas_gdf = gpd.GeoDataFrame(service_areas, crs=self.target_crs)
        
        # Create union of all service areas
        valid_geoms = [geom for geom in service_areas_gdf.geometry if geom and not geom.is_empty and geom.is_valid]
        
        if valid_geoms:
            unified_service_area = make_valid(unary_union(valid_geoms))
            self.served_areas = gpd.GeoDataFrame({'geometry': [unified_service_area]}, crs=self.target_crs)
        else:
            self.served_areas = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=self.target_crs), crs=self.target_crs)
        
        print(f"Created {len(service_areas_gdf)} service areas")
        return service_areas_gdf
    
    def identify_underserved_areas(self, min_area_threshold=1e-8):
        
        """
        Identify underserved areas within the boundaries.

        The underserved areas are identified as the difference between the unified
        boundaries and the unified service areas.

        Parameters:
        min_area_threshold : float, optional
            Minimum area threshold for an underserved area to be considered valid.
            Defaults to 1e-8.

        Returns:
            GeoDataFrame
            A GeoDataFrame containing the underserved areas with their geometry and
            area information.
        """
        if self.boundaries is None or self.served_areas is None:
            return None
        
        # Work with projected CRS
        boundaries_proj = self.boundaries.to_crs(self.projected_crs)
        served_areas_proj = self.served_areas.to_crs(self.projected_crs)
        
        unified_boundaries = make_valid(unary_union(boundaries_proj.geometry))
        
        if len(served_areas_proj) > 0 and not served_areas_proj.geometry.iloc[0].is_empty:
            unified_served = make_valid(unary_union(served_areas_proj.geometry))
            underserved_geometry = unified_boundaries.difference(unified_served)
        else:
            underserved_geometry = unified_boundaries
        
        underserved_polys = []
        
        # Handle geometry types
        if hasattr(underserved_geometry, 'geoms'):
            geoms = underserved_geometry.geoms
        else:
            geoms = [underserved_geometry]
            
        for geom in geoms:
            if (geom.is_valid and hasattr(geom, 'area') and 
                geom.area > min_area_threshold and
                geom.geom_type in ['Polygon', 'MultiPolygon']):
                underserved_polys.append({'geometry': geom})
        
        # Convert back to target CRS
        if underserved_polys:
            underserved_gdf = gpd.GeoDataFrame(underserved_polys, crs=self.projected_crs)
            self.underserved_areas = underserved_gdf.to_crs(self.target_crs)
        else:
            self.underserved_areas = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=self.target_crs), crs=self.target_crs)
        
        print(f"Identified {len(self.underserved_areas)} underserved areas")
        return self.underserved_areas
    
    def calculate_coverage_stats(self):
        """
        Calculate coverage statistics for the given set of boundaries and service areas.

        If self.boundaries is None, returns None.

        Returns a dictionary containing the following keys:
            - total_area: total area of the boundaries in square units of the projected CRS
            - served_area: total area of the served areas in square units of the projected CRS
            - coverage_percentage: percentage of the total area that is served
            - underserved_areas_count: number of underserved areas

        :return: coverage statistics or None
        """
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
    
    def run_analysis(self, poi_path, boundary_path, data_dir, output_dir, 
                    eps=None, buffer_distance=0.3, target_clusters=45):
        """
        Run a comprehensive service gap analysis on a set of input data.

        Loads input data from poi_path and boundary_path, clusters the points of interest, creates service areas, identifies underserved areas, and calculates coverage statistics.

        Args:
            poi_path (str): Path to the points of interest GeoJSON file.
            boundary_path (str): Path to the boundary GeoJSON file.
            data_dir (str): Directory containing input data files.
            output_dir (str): Directory to save output files.
            eps (float, optional): Epsilon value for DBSCAN clustering. Defaults to None.
            buffer_distance (float, optional): Distance to buffer service areas. Defaults to 0.3.
            target_clusters (int, optional): Target number of clusters to form. Defaults to 45.

        Returns:
            dict: A dictionary containing the results of the analysis, including the number of points processed, the number of clusters created, the number of served and underserved areas, and the coverage percentage.
        """
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
        
        # Save outputs
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