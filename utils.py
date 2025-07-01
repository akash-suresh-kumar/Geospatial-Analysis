import os
import geopandas as gpd

def get_data_dir():
    """Get the data directory relative to the current script location"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, 'data')

def get_output_dir():
    """Get the output directory relative to the current script location"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, 'Outputs')

def find_files_by_keywords(data_dir, keywords):
    """Find files matching certain keywords in the data directory"""
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist!")
        return []
    
    all_files = os.listdir(data_dir)
    matching_files = []
    
    for keyword in keywords:
        matches = [f for f in all_files if keyword.lower() in f.lower() and 
                  f.endswith(('.gpkg', '.geojson', '.shp', '.kml', '.gml'))]
        matching_files.extend(matches)
    
    return list(set(matching_files))

def list_spatial_files(data_dir):
    """List all available spatial files in the directory"""
    print(f"\nAvailable files in {data_dir}:")
    print("-" * 50)
    
    if not os.path.exists(data_dir):
        print("Directory does not exist!")
        return []
    
    spatial_extensions = ['.gpkg', '.geojson', '.shp', '.kml', '.gml']
    files = []
    
    for file in os.listdir(data_dir):
        if any(file.lower().endswith(ext) for ext in spatial_extensions):
            files.append(file)
            print(f"- {file}")
    
    print(f"\nTotal files found: {len(files)}")
    return files

def create_data_summary(data_dir):
    """Create a summary of all available spatial data"""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print("Data directory does not exist!")
        return
    
    spatial_extensions = ['.gpkg', '.geojson', '.shp', '.kml', '.gml']
    
    print(f"Data Directory: {data_dir}")
    print(f"{'File Name':<40} {'Type':<12} {'Count':<8} {'CRS':<15}")
    print("-" * 80)
    
    total_files = 0
    total_features = 0
    
    for file in os.listdir(data_dir):
        if any(file.lower().endswith(ext) for ext in spatial_extensions):
            try:
                file_path = os.path.join(data_dir, file)
                gdf = gpd.read_file(file_path)
                
                geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else "Unknown"
                count = len(gdf)
                crs = str(gdf.crs) if gdf.crs else "No CRS"
                
                print(f"{file:<40} {geom_type:<12} {count:<8} {crs:<15}")
                
                total_files += 1
                total_features += count
                
            except Exception as e:
                print(f"{file:<40} {'ERROR':<12} {str(e)[:20]:<20}")
    
    print("-" * 80)
    print(f"Total Files: {total_files}")
    print(f"Total Features: {total_features}")

def test_clustering_parameters(analyzer, file_paths, data_dir, test_output_dir):
    """Test different clustering parameters to find optimal settings"""
    print("\n" + "=" * 60)
    print("PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    eps_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    min_samples_values = [2, 3, 5]
    
    best_result = None
    best_params = None
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            print(f"Testing eps={eps}, min_samples={min_samples}")
            
            output_dir = os.path.join(test_output_dir, f"test_eps{eps}_ms{min_samples}")
            results = analyzer.run_analysis(file_paths, data_dir, output_dir, eps, min_samples)
            
            if results and results['clustered_points'] > 0:
                cluster_ratio = results['clustered_points'] / results['total_points']
                print(f"  → {cluster_ratio:.2%} points clustered, {results['hotspot_polygons']} hotspots")
                
                if 0.2 <= cluster_ratio <= 0.8:
                    if best_result is None or results['hotspot_polygons'] > best_result['hotspot_polygons']:
                        best_result = results
                        best_params = {'eps': eps, 'min_samples': min_samples}
            else:
                print("  → No clusters found")
    
    if best_params:
        print(f"\n RECOMMENDED PARAMETERS:")
        print(f"   eps = {best_params['eps']}")
        print(f"   min_samples = {best_params['min_samples']}")
    else:
        print("\n No optimal parameters found")
    
    return best_params