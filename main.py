import os
import sys
import glob

# Simple utility functions to replace utils.py
def get_data_dir():
    """
    Get the data directory relative to the current working directory.

    Returns:
        str: The full path to the 'data' directory.
    """
    return os.path.join(os.getcwd(), 'data')

def get_output_dir():
    """
    Get the output directory relative to the current working directory.

    Returns:
        str: The full path to the 'output' directory.
    """

    return os.path.join(os.getcwd(), 'output')

def find_files_by_keywords(data_dir, keywords):
    """
    Find files in the given data directory that have at least one of the given keywords in their filenames.

    Parameters:
        data_dir (str): The directory to search for files.
        keywords (list[str]): The keywords to search for.

    Returns:
        list[str]: A list of file names (not full paths) that matched the given keywords.
    """
    files = []
    for ext in ['*.shp', '*.geojson', '*.json', '*.gpkg']:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    matching_files = []
    for file in files:
        filename = os.path.basename(file).lower()
        if any(keyword.lower() in filename for keyword in keywords):
            matching_files.append(os.path.basename(file))
    
    return matching_files

def list_spatial_files(data_dir):
    files = []
    for ext in ['*.shp', '*.geojson', '*.json', '*.gpkg']:
        files.extend([os.path.basename(f) for f in glob.glob(os.path.join(data_dir, ext))])
    return files

# Import analyzers (assuming they're in the same directory)
try:
    from tourism_hotspot import TourismHotspotAnalyzer
    from service_gap_analyzer import ServiceGapAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure tourism_hotspot.py and service_gap_analyzer.py are in the same directory")
    sys.exit(1)

def run_comprehensive_tourism_analysis(num_clusters=25):
    """
    Run a comprehensive tourism analysis on all relevant spatial files in the data directory.

    This function identifies tourism-related files in the data directory, loads and processes them, 
    performs clustering to identify tourism hotspots, and generates detailed outputs including 
    hotspot polygons and a heatmap raster.

    Args:
        num_clusters (int, optional): The number of clusters to form. Defaults to 25.

    Returns:
        dict: A dictionary containing the results of the analysis, including the total number of points,
        clusters created, and hotspot polygons.
    """

    data_dir = get_data_dir()
    output_dir = os.path.join(get_output_dir(), 'comprehensive_tourism_analysis')
    
    # Find tourism-related files
    tourism_keywords = ['zoo', 'wildlife', 'sanctuary', 'mountain', 'tourism', 'tourist', 'temple', 'park']
    tourism_files = find_files_by_keywords(data_dir, tourism_keywords)
    
    if not tourism_files:
        tourism_files = list_spatial_files(data_dir)[:3]
    
    if not tourism_files:
        print("No spatial files found for tourism analysis!")
        return None
    
    # Initialize and run analysis
    analyzer = TourismHotspotAnalyzer(target_crs='EPSG:4326')
    
    results = analyzer.run_analysis(
        file_paths=tourism_files,
        data_dir=data_dir,
        output_dir=output_dir,
        n_clusters=num_clusters
    )
    
    # Manual detailed analysis
    detailed_output = os.path.join(output_dir, 'detailed')
    os.makedirs(detailed_output, exist_ok=True)
    
    # Load, cluster, and create polygons
    analyzer.load_vector_data(tourism_files, data_dir)
    analyzer.perform_clustering(n_clusters=num_clusters)
    analyzer.create_hotspot_polygons(buffer_distance=0.3)
    
    # Save detailed outputs
    try:
        if analyzer.clusters is not None:
            analyzer.clusters.to_file(os.path.join(detailed_output, 'detailed_clusters.geojson'))
        if analyzer.hotspot_polygons is not None:
            analyzer.hotspot_polygons.to_file(os.path.join(detailed_output, 'detailed_hotspots.geojson'))
        analyzer.generate_heatmap_raster(os.path.join(detailed_output, 'detailed_heatmap.tiff'), resolution=0.02)
    except Exception as e:
        print(f"Error saving detailed outputs: {e}")
    
    return results

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
    output_dir = os.path.join(get_output_dir(), 'comprehensive_service_gap_analysis')
    
    # Find service and boundary files
    service_keywords = ['atm', 'bank', 'post', 'school', 'hospital', 'service', 'poi', 'clinic', 'pharmacy']
    boundary_keywords = ['village', 'boundary', 'district', 'area', 'polygon', 'admin', 'ward']
    
    service_files = find_files_by_keywords(data_dir, service_keywords)
    boundary_files = find_files_by_keywords(data_dir, boundary_keywords)
    
    if not service_files or not boundary_files:
        print("Insufficient files for service gap analysis!")
        return None
    
    # Initialize and run analysis
    analyzer = ServiceGapAnalyzer(target_crs='EPSG:4326')
    
    results = analyzer.run_analysis(
        poi_path=service_files[0],
        boundary_path=boundary_files[0],
        data_dir=data_dir,
        output_dir=output_dir,
        eps=0.01,
        buffer_distance=0.05,
        target_clusters=target_clusters
    )
    
    # Manual detailed analysis
    detailed_output = os.path.join(output_dir, 'detailed')
    os.makedirs(detailed_output, exist_ok=True)
    
    # Load, cluster, and analyze
    analyzer.load_data(service_files[0], boundary_files[0], data_dir)
    poi_clusters = analyzer.cluster_pois(target_clusters=target_clusters)
    service_areas = analyzer.create_service_areas(poi_clusters, buffer_distance=0.05)
    analyzer.identify_underserved_areas()
    stats = analyzer.calculate_coverage_stats()
    
    # Save detailed outputs
    try:
        if poi_clusters is not None:
            poi_clusters.to_file(os.path.join(detailed_output, 'detailed_poi_clusters.geojson'))
        if service_areas is not None:
            service_areas.to_file(os.path.join(detailed_output, 'detailed_service_areas.geojson'))
        if analyzer.served_areas is not None and len(analyzer.served_areas) > 0:
            analyzer.served_areas.to_file(os.path.join(detailed_output, 'detailed_served_areas.geojson'))
        if analyzer.underserved_areas is not None and len(analyzer.underserved_areas) > 0:
            analyzer.underserved_areas.to_file(os.path.join(detailed_output, 'detailed_underserved_areas.geojson'))
    except Exception as e:
        print(f"Error saving detailed outputs: {e}")
    
    return results

def main():
    # Check if data directory exists
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
    
    # List available files
    spatial_files = list_spatial_files(data_dir)
    if not spatial_files:
        print("No spatial files found in data directory!")
        return
    
    print(f"Found {len(spatial_files)} spatial files")
    
    # Run analyses
    print("Choose analysis type:")
    print("1. Tourism Hotspot Analysis")
    print("2. Service Gap Analysis")
    print("3. Both Analyses")
    
    try:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            clusters = int(input("Enter number of clusters for tourism analysis (default 25): ") or "25")
            results = run_comprehensive_tourism_analysis(num_clusters=clusters)
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
            
            tourism_results = run_comprehensive_tourism_analysis(num_clusters=tourism_clusters)
            service_results = run_comprehensive_service_gap_analysis(target_clusters=service_clusters)
            
            if tourism_results:
                print(f"Tourism analysis: {tourism_results['total_points']} points, {tourism_results['clusters_created']} clusters")
            if service_results:
                print(f"Service analysis: {service_results['coverage_percentage']:.1f}% coverage, {service_results['underserved_areas_count']} underserved areas")
        
        else:
            print("Invalid choice!")
            return
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error during analysis: {e}")
    
    print("Analysis complete! Check the output directory for results.")

if __name__ == "__main__":
    main()