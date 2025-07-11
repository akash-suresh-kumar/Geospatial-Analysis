import os
import glob
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point
from shapely.ops import unary_union

warnings.filterwarnings('ignore')

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

def load_vector_data(file_paths, data_dir, target_crs='EPSG:4326'):
    """Load and merge vector data from multiple files"""
    gdfs = []
    for file_path in file_paths:
        try:
            full_path = os.path.join(data_dir, file_path) if not os.path.isabs(file_path) else file_path
            gdf = gpd.read_file(full_path)
            
            # Convert non-point geometries to centroids
            if gdf.geometry.geom_type.iloc[0] not in ['Point', 'MultiPoint']:
                gdf.geometry = gdf.geometry.centroid
            
            # Reproject if necessary
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
            
            gdf['source'] = os.path.basename(full_path)
            gdfs.append(gdf)
            print(f"Loaded: {os.path.basename(full_path)} ({len(gdf)} points)")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if gdfs:
        merged_data = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        merged_data.crs = target_crs
        print(f"Total points loaded: {len(merged_data)}")
        return merged_data
    else:
        return gpd.GeoDataFrame()

def perform_clustering(data, n_clusters=25):
    """Perform K-means clustering on point data"""
    if data is None or len(data) == 0:
        return None
    
    coords = np.array([[p.x, p.y] for p in data.geometry])
    actual_clusters = min(n_clusters, len(data))
    
    labels = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10).fit_predict(coords)
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = labels
    
    cluster_counts = pd.Series(labels).value_counts()
    print(f"Created {actual_clusters} clusters from {len(data)} points")
    print(f"Points per cluster: min={cluster_counts.min()}, max={cluster_counts.max()}, avg={cluster_counts.mean():.1f}")
    
    return data_with_clusters

def create_hotspot_polygons(clustered_data, buffer_distance=0.3, target_crs='EPSG:4326'):
    """Create hotspot polygons from clustered points"""
    if clustered_data is None or len(clustered_data) == 0:
        return None
    
    hotspot_polygons = []
    for cluster_id in clustered_data['cluster'].unique():
        cluster_points = clustered_data[clustered_data['cluster'] == cluster_id]
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
    
    hotspot_gdf = gpd.GeoDataFrame(hotspot_polygons, crs=target_crs)
    print(f"Created {len(hotspot_gdf)} hotspot polygons")
    return hotspot_gdf

def generate_heatmap_raster(data, output_path, resolution=0.05, target_crs='EPSG:4326'):
    """Generate a heatmap raster from point data"""
    if data is None or len(data) == 0:
        return
    
    coords = np.array([[p.x, p.y] for p in data.geometry])
    
    # Calculate bounds with buffer
    buffer = 1.0
    xmin, ymin = coords.min(axis=0) - buffer
    xmax, ymax = coords.max(axis=0) + buffer
    
    # Create grid
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)
    
    x_range = np.linspace(xmin, xmax, width)
    y_range = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate density
    positions = np.vstack([X.ravel(), Y.ravel()])
    density = gaussian_kde(coords.T)(positions).reshape(X.shape)
    density = np.flipud(density)  # Flip to match raster orientation
    
    # Create transform
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    
    # Write raster
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=density.dtype,
        crs=target_crs,
        transform=transform
    ) as dst:
        dst.write(density, 1)
    
    print(f"Heatmap saved: {output_path}")

def run_tourism_analysis(file_paths=None, n_clusters=25, create_heatmap=True, resolution=0.05, target_crs='EPSG:4326'):
    """Run complete tourism hotspot analysis"""
    data_dir = get_data_dir()
    output_dir = os.path.join(get_output_dir(), 'tourism_analysis')
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please create the data directory and add your spatial files.")
        return None
    
    # Find tourism-related files if not provided
    if file_paths is None:
        tourism_keywords = ['zoo', 'wildlife', 'sanctuary', 'mountain', 'tourism', 'tourist', 'temple', 'park']
        file_paths = find_files_by_keywords(data_dir, tourism_keywords)
        
        if not file_paths:
            file_paths = list_spatial_files(data_dir)[:3]  # Take first 3 files
    
    if not file_paths:
        print("No spatial files found for tourism analysis!")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    merged_data = load_vector_data(file_paths, data_dir, target_crs)
    if merged_data is None or len(merged_data) == 0:
        print("No valid data loaded.")
        return None
    
    # Perform clustering
    clustered_data = perform_clustering(merged_data, n_clusters)
    if clustered_data is None:
        print("Clustering failed.")
        return None
    
    # Create hotspot polygons
    hotspot_polygons = create_hotspot_polygons(clustered_data, target_crs=target_crs)
    
    # Generate outputs
    try:
        # Save clustered points
        if clustered_data is not None:
            clustered_data.to_file(os.path.join(output_dir, 'clustered_points.geojson'))
            print("Saved clustered points")
        
        # Save hotspot polygons
        if hotspot_polygons is not None:
            hotspot_polygons.to_file(os.path.join(output_dir, 'tourism_hotspots.geojson'))
            print("Saved tourism hotspots")
        
        # Generate heatmap
        if create_heatmap:
            generate_heatmap_raster(
                merged_data, 
                os.path.join(output_dir, 'tourism_heatmap.tiff'), 
                resolution, 
                target_crs
            )
    
    except Exception as e:
        print(f"Error saving outputs: {e}")
    
    print(f"\nTourism analysis complete! Output: {output_dir}")
    
    return {
        'total_points': len(merged_data),
        'clusters_created': len(clustered_data['cluster'].unique()) if clustered_data is not None else 0,
        'hotspot_polygons': len(hotspot_polygons) if hotspot_polygons is not None else 0,
        'output_dir': output_dir
    }

def main():
    """Main entry point for running tourism hotspot analysis
    
    This function runs a comprehensive tourism hotspot analysis on the given files.
    It loads and processes the given files, performs clustering to identify
    tourism hotspots, and generates detailed outputs including hotspot polygons
    and a heatmap raster.
    
    Parameters:
        n_clusters (int): The number of clusters to form. Defaults to 25.
        create_heatmap (bool): Generate service accessibility heatmap? Defaults to True.
        resolution (float): Resolution of the heatmap. Defaults to 0.05.
    
    Returns:
        dict: A dictionary containing the results of the analysis, including the
            total number of points, clusters created, and hotspot polygons.
    """
    """Main function to run tourism hotspot analysis"""
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
        # Get user input
        n_clusters = int(input("Enter number of clusters (default 25): ") or "25")
        create_heatmap = input("Create heatmap? (y/n, default: y): ").strip().lower() in ['', 'y', 'yes']
        resolution = float(input("Enter resolution for heatmap (default: 0.05): ") or "0.05")
        
        # Run analysis
        results = run_tourism_analysis(
            n_clusters=n_clusters,
            create_heatmap=create_heatmap,
            resolution=resolution
        )
        
        if results:
            print(f"\nAnalysis Results:")
            print(f"- Total points processed: {results['total_points']}")
            print(f"- Clusters created: {results['clusters_created']}")
            print(f"- Hotspot polygons: {results['hotspot_polygons']}")
            print(f"- Output directory: {results['output_dir']}")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()