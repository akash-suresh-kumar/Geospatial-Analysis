import os
from Tourism_hotspot import (
    TourismHotspotAnalyzer, ServiceGapAnalyzer, 
    get_data_dir, get_output_dir, find_files_by_keywords, list_spatial_files
)

def run_comprehensive_tourism_analysis(num_clusters=25, create_heatmap=True, resolution=0.05):
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

# TODO: Add FacilityLocationOptimizer class here
# This would include methods for:
# - Location optimization using techniques like p-median, p-center, or capacitated facility location
# - Demand point analysis
# - Cost-distance modeling
# - Multi-criteria optimization
# - Accessibility analysis

def main():
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