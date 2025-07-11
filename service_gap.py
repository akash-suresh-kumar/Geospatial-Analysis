import os
import glob
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

try:
    from shapely import make_valid
except ImportError:
    make_valid = lambda geom: geom.buffer(0) if geom and not geom.is_empty else geom

def get_data_dir():
    return os.path.join(os.getcwd(), 'data')

def get_output_dir():
    return os.path.join(os.getcwd(), 'output')

def find_files_by_keywords(data_dir, keywords):
    """Find spatial files containing specific keywords"""
    files = []
    for ext in ['*.shp', '*.geojson', '*.json', '*.gpkg']:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    return [os.path.basename(f) for f in files if any(k.lower() in os.path.basename(f).lower() for k in keywords)]

def list_spatial_files(data_dir):
    """List all spatial files in the data directory"""
    files = []
    for ext in ['*.shp', '*.geojson', '*.json', '*.gpkg']:
        files.extend([os.path.basename(f) for f in glob.glob(os.path.join(data_dir, ext))])
    return files

def get_projected_crs(gdf):
    """Get appropriate projected CRS for the given GeoDataFrame"""
    if gdf.crs is None or gdf.empty:
        return 'EPSG:4326'
    
    centroid = gdf.geometry.centroid.iloc[0]
    utm_zone = int((centroid.x + 180) / 6) + 1
    hemisphere = 'north' if centroid.y >= 0 else 'south'
    
    if hemisphere == 'north':
        return f'EPSG:326{utm_zone:02d}'
    else:
        return f'EPSG:327{utm_zone:02d}'

def load_poi_data(poi_path, data_dir, target_crs='EPSG:4326'):
    """Load Points of Interest data"""
    try:
        poi_full_path = os.path.join(data_dir, poi_path) if not os.path.isabs(poi_path) else poi_path
        pois = gpd.read_file(poi_full_path)
        
        # Reproject if necessary
        if pois.crs != target_crs:
            pois = pois.to_crs(target_crs)
        
        # Convert to points if necessary
        if not all(pois.geometry.geom_type == 'Point'):
            pois.geometry = pois.geometry.centroid
        
        print(f"Loaded {len(pois)} POIs from {os.path.basename(poi_path)}")
        return pois
    except Exception as e:
        print(f"Error loading POI file {poi_path}: {e}")
        return None

def load_boundary_data(boundary_path, data_dir, target_crs='EPSG:4326'):
    """Load boundary data"""
    try:
        boundary_full_path = os.path.join(data_dir, boundary_path) if not os.path.isabs(boundary_path) else boundary_path
        boundaries = gpd.read_file(boundary_full_path)
        
        # Reproject if necessary
        if boundaries.crs != target_crs:
            boundaries = boundaries.to_crs(target_crs)
        
        print(f"Loaded {len(boundaries)} boundaries from {os.path.basename(boundary_path)}")
        return boundaries
    except Exception as e:
        print(f"Error loading boundary file {boundary_path}: {e}")
        return None

def find_optimal_eps(pois, target_clusters=45, k=4):
    """Find optimal epsilon value for DBSCAN clustering"""
    if pois is None or len(pois) == 0:
        return 0.01
    
    coords = np.array([[p.x, p.y] for p in pois.geometry])
    
    # Calculate k-nearest neighbors distances
    neighbors = NearestNeighbors(n_neighbors=min(k, len(coords)))
    distances, _ = neighbors.fit(coords).kneighbors(coords)
    
    # Try different percentiles of the k-th nearest neighbor distances
    eps_candidates = np.percentile(np.sort(distances[:, -1]), [50, 60, 70, 80, 85, 90, 95])
    
    best_eps = 0.01
    best_cluster_count = 0
    
    for eps in eps_candidates:
        labels = DBSCAN(eps=eps, min_samples=2).fit_predict(coords)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if abs(n_clusters - target_clusters) < abs(best_cluster_count - target_clusters):
            best_eps = eps
            best_cluster_count = n_clusters
    
    print(f"Selected eps={best_eps:.4f} for ~{best_cluster_count} clusters")
    return best_eps

def cluster_pois(pois, eps=None, min_samples=2, target_clusters=45):
    """Cluster POIs using DBSCAN algorithm"""
    if pois is None or len(pois) == 0:
        print("No POIs available for clustering")
        return None
    
    # Find optimal eps if not provided
    if eps is None:
        eps = find_optimal_eps(pois, target_clusters=target_clusters)
    
    coords = np.array([[p.x, p.y] for p in pois.geometry])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    
    # Create clustered data
    clustered_pois = pois.copy()
    clustered_pois['cluster'] = labels
    
    # Separate clusters from outliers
    poi_clusters = clustered_pois[clustered_pois['cluster'] != -1].copy()
    outliers = clustered_pois[clustered_pois['cluster'] == -1].copy()
    
    # Assign individual cluster IDs to outliers
    if len(outliers) > 0:
        max_cluster = poi_clusters['cluster'].max() if len(poi_clusters) > 0 else -1
        outliers['cluster'] = range(max_cluster + 1, max_cluster + 1 + len(outliers))
        poi_clusters = gpd.GeoDataFrame(
            pd.concat([poi_clusters, outliers], ignore_index=True), 
            crs=pois.crs
        )
    
    n_real_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = sum(1 for x in labels if x == -1)
    print(f"Clustering results: {n_real_clusters} clusters, {n_outliers} individual POIs")
    
    return poi_clusters

def create_service_areas(poi_clusters, buffer_distance=0.3, target_crs='EPSG:4326'):
    """Create service areas from clustered POIs"""
    if poi_clusters is None or len(poi_clusters) == 0:
        return None, None
    
    # Create individual service areas for each cluster
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
    
    service_areas_gdf = gpd.GeoDataFrame(service_areas, crs=target_crs)
    
    # Create unified service area
    valid_geoms = [geom for geom in service_areas_gdf.geometry 
                   if geom and not geom.is_empty and geom.is_valid]
    
    if valid_geoms:
        unified_service_area = make_valid(unary_union(valid_geoms))
        served_areas = gpd.GeoDataFrame(
            {'geometry': [unified_service_area]}, 
            crs=target_crs
        )
    else:
        served_areas = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([], crs=target_crs), 
            crs=target_crs
        )
    
    print(f"Created {len(service_areas_gdf)} service areas")
    return service_areas_gdf, served_areas

def identify_underserved_areas(boundaries, served_areas, target_crs='EPSG:4326', min_area_threshold=1e-8):
    """Identify underserved areas by subtracting served areas from boundaries"""
    if boundaries is None or served_areas is None:
        return None
    
    # Get projected CRS for area calculations
    projected_crs = get_projected_crs(boundaries)
    
    # Project to local CRS for accurate area calculations
    boundaries_proj = boundaries.to_crs(projected_crs)
    served_areas_proj = served_areas.to_crs(projected_crs)
    
    # Create unified boundary
    unified_boundaries = make_valid(unary_union(boundaries_proj.geometry))
    
    # Calculate underserved areas
    if len(served_areas_proj) > 0 and not served_areas_proj.geometry.iloc[0].is_empty:
        unified_served = make_valid(unary_union(served_areas_proj.geometry))
        underserved_geometry = unified_boundaries.difference(unified_served)
    else:
        underserved_geometry = unified_boundaries
    
    # Extract individual polygons
    underserved_polys = []
    if hasattr(underserved_geometry, 'geoms'):
        geoms = underserved_geometry.geoms
    else:
        geoms = [underserved_geometry]
    
    for geom in geoms:
        if (geom.is_valid and 
            hasattr(geom, 'area') and 
            geom.area > min_area_threshold and 
            geom.geom_type in ['Polygon', 'MultiPolygon']):
            underserved_polys.append({'geometry': geom})
    
    if underserved_polys:
        underserved_gdf = gpd.GeoDataFrame(underserved_polys, crs=projected_crs)
        underserved_areas = underserved_gdf.to_crs(target_crs)
    else:
        underserved_areas = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([], crs=target_crs), 
            crs=target_crs
        )
    
    print(f"Identified {len(underserved_areas)} underserved areas")
    return underserved_areas

def calculate_coverage_stats(boundaries, served_areas):
    """Calculate coverage statistics"""
    if boundaries is None:
        return None
    
    # Get projected CRS for accurate area calculations
    projected_crs = get_projected_crs(boundaries)
    
    # Project to local CRS
    boundaries_projected = boundaries.to_crs(projected_crs)
    total_area = make_valid(unary_union(boundaries_projected.geometry)).area
    
    served_area = 0
    if served_areas is not None and len(served_areas) > 0:
        served_areas_projected = served_areas.to_crs(projected_crs)
        if not served_areas_projected.geometry.iloc[0].is_empty:
            served_area = make_valid(unary_union(served_areas_projected.geometry)).area
    
    coverage_pct = min((served_area / total_area * 100) if total_area > 0 else 0, 100.0)
    
    return {
        'total_area': total_area,
        'served_area': served_area,
        'coverage_percentage': coverage_pct
    }

def create_service_heatmap(boundaries, poi_clusters, resolution=100, output_dir=None):
    """Create a heatmap showing service accessibility"""
    if boundaries is None or poi_clusters is None:
        print("Cannot create heatmap: missing boundary or POI data")
        return
    
    # Get bounds
    bounds = boundaries.total_bounds
    minx, miny, maxx, maxy = bounds
    
    # Create grid
    x = np.linspace(minx, maxx, resolution)
    y = np.linspace(miny, maxy, resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Create grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Calculate distances to nearest POI for each grid point
    poi_coords = np.array([[p.x, p.y] for p in poi_clusters.geometry])
    distances = cdist(grid_points, poi_coords).min(axis=1)
    
    # Reshape to grid
    distance_grid = distances.reshape(xx.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create custom colormap (green = well served, red = underserved)
    colors = ['#d73027', '#fc8d59', '#fee08b', '#e6f598', '#99d594', '#3288bd']
    cmap = LinearSegmentedColormap.from_list('service_access', colors[::-1])
    
    # Plot heatmap
    im = ax.contourf(xx, yy, distance_grid, levels=20, cmap=cmap, alpha=0.7)
    
    # Add boundaries
    boundaries.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.8)
    
    # Add POIs
    poi_clusters.plot(ax=ax, color='red', markersize=10, alpha=0.8, label='Service Points')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Distance to Nearest Service (degrees)', rotation=270, labelpad=20)
    
    # Set title and labels
    ax.set_title('Service Accessibility Heatmap', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    
    # Set aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        heatmap_path = os.path.join(output_dir, 'service_accessibility_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {heatmap_path}")
    
    plt.show()

def run_service_gap_analysis(poi_path=None, boundary_path=None, eps=None, buffer_distance=0.3, 
                           target_clusters=45, target_crs='EPSG:4326', generate_heatmap=False, 
                           heatmap_resolution=100):
    """Run complete service gap analysis"""
    data_dir = get_data_dir()
    output_dir = os.path.join(get_output_dir(), 'service_gap_analysis')
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please create the data directory and add your spatial files.")
        return None
    
    # Find service and boundary files if not provided
    if poi_path is None or boundary_path is None:
        service_keywords = ['atm', 'bank', 'post', 'school', 'hospital', 'service', 'poi', 'clinic', 'pharmacy']
        boundary_keywords = ['village', 'boundary', 'district', 'area', 'polygon', 'admin', 'ward']
        
        service_files = find_files_by_keywords(data_dir, service_keywords)
        boundary_files = find_files_by_keywords(data_dir, boundary_keywords)
        
        if not service_files or not boundary_files:
            all_files = list_spatial_files(data_dir)
            print("Available files:")
            for i, file in enumerate(all_files):
                print(f"  {i+1}. {file}")
            
            if not service_files:
                print("No service files found automatically. Please specify POI file.")
                return None
            if not boundary_files:
                print("No boundary files found automatically. Please specify boundary file.")
                return None
        
        poi_path = service_files[0]
        boundary_path = boundary_files[0]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    pois = load_poi_data(poi_path, data_dir, target_crs)
    boundaries = load_boundary_data(boundary_path, data_dir, target_crs)
    
    if pois is None or boundaries is None:
        print("Failed to load required data files.")
        return None
    
    # Cluster POIs
    poi_clusters = cluster_pois(pois, eps=eps, target_clusters=target_clusters)
    if poi_clusters is None:
        print("POI clustering failed.")
        return None
    
    # Create service areas
    service_areas, served_areas = create_service_areas(poi_clusters, buffer_distance, target_crs)
    if service_areas is None:
        print("Service area creation failed.")
        return None
    
    # Identify underserved areas
    underserved_areas = identify_underserved_areas(boundaries, served_areas, target_crs)
    
    # Calculate coverage statistics
    stats = calculate_coverage_stats(boundaries, served_areas)
    if stats:
        stats['underserved_areas_count'] = len(underserved_areas) if underserved_areas is not None else 0
        stats['poi_count'] = len(pois)
        stats['cluster_count'] = len(poi_clusters['cluster'].unique()) if poi_clusters is not None else 0
    
    # Generate heatmap if requested
    if generate_heatmap:
        print("Generating service accessibility heatmap...")
        create_service_heatmap(boundaries, poi_clusters, heatmap_resolution, output_dir)
    
    # Save outputs
    try:
        # Save POI clusters
        if poi_clusters is not None and len(poi_clusters) > 0:
            poi_clusters.to_file(os.path.join(output_dir, 'poi_clusters.geojson'))
            print("Saved POI clusters")
        
        # Save service areas
        if service_areas is not None and len(service_areas) > 0:
            service_areas.to_file(os.path.join(output_dir, 'service_areas.geojson'))
            print("Saved service areas")
        
        # Save served areas
        if served_areas is not None and len(served_areas) > 0:
            served_areas.to_file(os.path.join(output_dir, 'well_served_areas.geojson'))
            print("Saved well-served areas")
        
        # Save underserved areas
        if underserved_areas is not None and len(underserved_areas) > 0:
            underserved_areas.to_file(os.path.join(output_dir, 'underserved_areas.geojson'))
            print("Saved underserved areas")
    
    except Exception as e:
        print(f"Error saving outputs: {e}")
    
    if stats:
        print(f"\nSERVICE GAP ANALYSIS COMPLETE!")
        print(f"POIs processed: {stats['poi_count']}")
        print(f"Clusters created: {stats['cluster_count']}")
        print(f"Coverage: {stats['coverage_percentage']:.1f}%")
        print(f"Underserved areas: {stats['underserved_areas_count']}")
        print(f"Output directory: {output_dir}")
    
    return stats

def get_user_input():
    """Get user input for analysis parameters"""
    print("=== SERVICE GAP ANALYSIS CONFIGURATION ===")
    
    # Get target clusters
    while True:
        try:
            target_clusters = input("Enter target number of clusters (default 45): ").strip()
            target_clusters = int(target_clusters) if target_clusters else 45
            if target_clusters > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get buffer distance
    while True:
        try:
            buffer_distance = input("Enter buffer distance for service areas (default 0.3): ").strip()
            buffer_distance = float(buffer_distance) if buffer_distance else 0.3
            if buffer_distance > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Ask about heatmap generation
    while True:
        heatmap_choice = input("Generate service accessibility heatmap? (y/n, default n): ").strip().lower()
        if heatmap_choice in ['', 'n', 'no']:
            generate_heatmap = False
            heatmap_resolution = 100
            break
        elif heatmap_choice in ['y', 'yes']:
            generate_heatmap = True
            # Get heatmap resolution
            while True:
                try:
                    resolution = input("Enter heatmap resolution (default 100): ").strip()
                    heatmap_resolution = int(resolution) if resolution else 100
                    if heatmap_resolution > 0:
                        break
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Please enter a valid number.")
            break
        else:
            print("Please enter 'y' or 'n'.")
    
    return target_clusters, buffer_distance, generate_heatmap, heatmap_resolution

def main():
    """Main function to run service gap analysis"""
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
    print("Available files:")
    for i, file in enumerate(spatial_files):
        print(f"  {i+1}. {file}")
    
    try:
        # Get file input
        poi_file = input("Enter POI filename (or press Enter for auto-detection): ").strip()
        boundary_file = input("Enter boundary filename (or press Enter for auto-detection): ").strip()
        
        poi_path = poi_file if poi_file else None
        boundary_path = boundary_file if boundary_file else None
        
        # Get analysis parameters
        target_clusters, buffer_distance, generate_heatmap, heatmap_resolution = get_user_input()
        
        # Run analysis
        results = run_service_gap_analysis(
            poi_path=poi_path,
            boundary_path=boundary_path,
            target_clusters=target_clusters,
            buffer_distance=buffer_distance,
            generate_heatmap=generate_heatmap,
            heatmap_resolution=heatmap_resolution
        )
        
        if results:
            print("\nAnalysis completed successfully!")
        else:
            print("\nAnalysis failed. Please check your input files and try again.")
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")

if __name__ == "__main__":
    main()