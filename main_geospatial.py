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
    """Find files in the given data directory that have at least one of the given keywords in their filenames.

    Parameters:
        data_dir (str): The directory to search for files.
        keywords (list[str]): The keywords to search for.

    Returns:
        list[str]: A list of file names (not full paths) that matched the given keywords.
    """
    files = []
    for ext in ['*.shp', '*.geojson', '*.json', '*.gpkg']:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    return [os.path.basename(f) for f in files if any(k.lower() in os.path.basename(f).lower() for k in keywords)]

def list_spatial_files(data_dir):
    """List all spatial files in the given data directory.

    Parameters:
        data_dir (str): The directory to search for files.

    Returns:
        list[str]: A list of file names (not full paths) of all spatial files.
    """
    files = []
    for ext in ['*.shp', '*.geojson', '*.json', '*.gpkg']:
        files.extend([os.path.basename(f) for f in glob.glob(os.path.join(data_dir, ext))])
    return files

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
        self.merged_data = self.clusters = self.hotspot_polygons = None
        
    def load_vector_data(self, file_paths, data_dir):
        """
        Load vector data from files and merge into a single GeoDataFrame.

        Parameters
        ----------
        file_paths : list[str]
            List of file paths to load data from.
        data_dir : str
            Directory containing the files to load.

        Returns
        -------
        geopandas.GeoDataFrame
            Merged GeoDataFrame of all data loaded from files.
        """
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
        """
        Create polygons around clusters of points, buffering each cluster by a set distance.

        Parameters
        ----------
        buffer_distance : float, optional
            Distance to buffer each cluster's convex hull by. Defaults to 0.3.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame of tourism hotspot polygons.
        """
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
        """
        Run a comprehensive tourism analysis on the given files.

        This function loads and processes the given files, performs clustering to identify
        tourism hotspots, and generates detailed outputs including hotspot polygons and a
        heatmap raster.

        Parameters:
            file_paths (list[str]):
                A list of paths to the spatial files to be analyzed.
            data_dir (str):
                The directory containing the spatial files.
            output_dir (str):
                The directory to save the outputs to.
            n_clusters (int, optional):
                The number of clusters to form. Defaults to 25.
            create_heatmap (bool, optional):
                Whether to generate a heatmap raster of the points. Defaults to True.
            resolution (float, optional):
                The resolution of the heatmap raster in the units of the target CRS.
                Defaults to 0.05.

        Returns:
            dict:
                A dictionary containing the results of the analysis, including the total
                number of points, clusters created, and hotspot polygons.
        """
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
        """
        Initialize ServiceGapAnalyzer.

        Parameters
        ----------
        target_crs : str, optional
            Target coordinate reference system (CRS) of the data. Defaults to 'EPSG:4326' (WGS84 lat/lon).

        Attributes
        ----------
        target_crs : str
            Target coordinate reference system (CRS) of the data.
        pois : geopandas.GeoDataFrame or None
            GeoDataFrame of points of interest.
        boundaries : geopandas.GeoDataFrame or None
            GeoDataFrame of boundaries.
        served_areas : geopandas.GeoDataFrame or None
            GeoDataFrame of served areas.
        underserved_areas : geopandas.GeoDataFrame or None
            GeoDataFrame of underserved areas.
        projected_crs : str or None
            The projected CRS used for the analysis.
        """
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
            True if successful, False otherwise.
        """
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
        """
        Identify areas that are not served by any POI service area.
        
        Args:
            min_area_threshold (float, optional): Minimum area of an underserved area to be considered. Defaults to 1e-8.
        
        Returns:
            GeoDataFrame: A GeoDataFrame containing the underserved areas.
        """
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
    
    def run_analysis(self, poi_path, boundary_path, data_dir, output_dir, eps=None, buffer_distance=0.3, target_clusters=45):
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

def run_comprehensive_tourism_analysis(num_clusters=25, create_heatmap=True, resolution=0.05):
    """
    Run a comprehensive tourism analysis on relevant spatial files in the data directory.

    This function identifies tourism-related files based on specified keywords, loads
    and processes these files, performs clustering to identify tourism hotspots, and
    optionally generates a heatmap raster. The results are saved to the specified output
    directory.

    Args:
        num_clusters (int, optional): The number of clusters to form. Defaults to 25.
        create_heatmap (bool, optional): Whether to generate a heatmap raster of the points.
            Defaults to True.
        resolution (float, optional): The resolution of the heatmap raster in the units of the
            target CRS. Defaults to 0.05.

    Returns:
        dict or None: A dictionary containing the results of the analysis, including the total
        number of points, clusters created, and hotspot polygons, or None if no files are found.
    """

    data_dir = get_data_dir()
    output_dir = os.path.join(get_output_dir(), 'tourism_analysis')
    tourism_keywords = ['zoo', 'wildlife', 'sanctuary', 'mountain', 'tourism', 'tourist', 'temple', 'park']
    tourism_files = find_files_by_keywords(data_dir, tourism_keywords)
    
    if not tourism_files:
        tourism_files = list_spatial_files(data_dir)[:3]
    
    if not tourism_files:
        print("No spatial files found for tourism analysis!")
        return None
    
    analyzer = TourismHotspotAnalyzer(target_crs='EPSG:4326')
    return analyzer.run_analysis(tourism_files, data_dir, output_dir, num_clusters, create_heatmap, resolution)

def run_comprehensive_service_gap_analysis(target_clusters=45):
    """
    Run a comprehensive service gap analysis on all service and boundary files in data_dir.

    This function finds all service and boundary files in data_dir, loads the first service file and the first boundary file, clusters the points of interest, creates service areas, identifies underserved areas, and calculates coverage statistics.

    Args:
        target_clusters (int, optional): The target number of clusters to form. Defaults to 45.

    Returns:
        dict: A dictionary containing the results of the analysis, including the number of points processed, the number of clusters created, the number of served and underserved areas, and the coverage percentage.
    """
    data_dir = get_data_dir()
    output_dir = os.path.join(get_output_dir(), 'service_gap_analysis')
    service_keywords = ['atm', 'bank', 'post', 'school', 'hospital', 'service', 'poi', 'clinic', 'pharmacy']
    boundary_keywords = ['village', 'boundary', 'district', 'area', 'polygon', 'admin', 'ward']
    
    service_files = find_files_by_keywords(data_dir, service_keywords)
    boundary_files = find_files_by_keywords(data_dir, boundary_keywords)
    
    if not service_files or not boundary_files:
        print("Insufficient files for service gap analysis!")
        return None
    
    analyzer = ServiceGapAnalyzer(target_crs='EPSG:4326')
    return analyzer.run_analysis(service_files[0], boundary_files[0], data_dir, output_dir, 0.01, 0.05, target_clusters)

def main():
    """
    Main entry point for running comprehensive spatial analyses.

    This function checks if the data directory exists, lists available spatial files, and
    prompts the user to choose an analysis type. It then runs the chosen analysis and
    prints the results.

    If the user chooses to run both analyses, this function runs both in sequence and
    prints their results.

    If an error occurs during analysis, it is caught and an error message is printed.

    If the user interrupts the analysis with Ctrl+C, this function prints a message and
    exits.

    """
    data_dir = get_data_dir()
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please create the data directory and add your spatial files.")
        return
    
    spatial_files = list_spatial_files(data_dir)
    if not spatial_files:
        print("No spatial files found in data directory!")
        return
    
    print(f"Found {len(spatial_files)} spatial files")
    print("Choose analysis type:")
    print("1. Tourism Hotspot Analysis")
    print("2. Service Gap Analysis")
    print("3. Both Analyses")
    
    try:
        choice = input("Enter your choice (1-3): ").strip()
        create_heatmap = input("Create heatmaps? (y/n, default: y): ").strip().lower() in ['', 'y', 'yes']
        resolution = float(input("Enter resolution for heatmaps (default: 0.05): ") or "0.05")
        
        if choice == '1':
            clusters = int(input("Enter number of clusters for tourism analysis (default 25): ") or "25")
            results = run_comprehensive_tourism_analysis(num_clusters=clusters, create_heatmap=create_heatmap, resolution=resolution)
            if results:
                print(f"Tourism analysis complete. Processed {results['total_points']} points, created {results['clusters_created']} clusters")
        
        elif choice == '2':
            clusters = int(input("Enter target clusters for service gap analysis (default 45): ") or "45")
            results = run_comprehensive_service_gap_analysis(target_clusters=clusters)
            if results:
                print(f"Service gap analysis complete. Coverage: {results['coverage_percentage']:.1f}%, Underserved areas: {results['underserved_areas_count']}")
        
        elif choice == '3':
            tourism_clusters = int(input("Enter number of clusters for tourism analysis (default 25): ") or "25")
            service_clusters = int(input("Enter target clusters for service gap analysis (default 45): ") or "45")
            
            tourism_results = run_comprehensive_tourism_analysis(num_clusters=tourism_clusters, create_heatmap=create_heatmap, resolution=resolution)
            service_results = run_comprehensive_service_gap_analysis(target_clusters=service_clusters)
            
            if tourism_results:
                print(f"Tourism analysis: {tourism_results['total_points']} points, {tourism_results['clusters_created']} clusters")
            if service_results:
                print(f"Service gap analysis: Coverage: {service_results['coverage_percentage']:.1f}%, Underserved areas: {service_results['underserved_areas_count']}")
        else:
            print("Invalid choice. Please run the script again.")
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()