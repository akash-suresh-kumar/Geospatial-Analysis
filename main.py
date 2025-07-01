import os
import sys

try:
    from tourism_hotspot import TourismHotspotAnalyzer
    from service_gap_analyzer import ServiceGapAnalyzer
    from utils import get_data_dir, get_output_dir, find_files_by_keywords, list_spatial_files
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the same directory")
    sys.exit(1)

def run_service_gap_analysis():
    """Run service gap analysis"""
    print("\n" + "=" * 60)
    print("SERVICE GAP ANALYSIS")
    print("=" * 60)
    
    data_dir = get_data_dir()
    output_dir = os.path.join(get_output_dir(), 'service_gap_analysis')
    
    # Find service and boundary files
    service_keywords = ['atm', 'bank', 'post', 'school', 'hospital', 'service', 'poi']
    boundary_keywords = ['village', 'boundary', 'district', 'area', 'polygon']
    
    service_files = find_files_by_keywords(data_dir, service_keywords)
    boundary_files = find_files_by_keywords(data_dir, boundary_keywords)
    
    print(f"Service files: {service_files}")
    print(f"Boundary files: {boundary_files}")
    
    if not service_files or not boundary_files:
        print("Insufficient files for service gap analysis!")
        return None
    
    analyzer = ServiceGapAnalyzer()
    results = analyzer.run_analysis(
        poi_path=service_files[0],
        boundary_path=boundary_files[0],
        data_dir=data_dir,
        output_dir=output_dir,
        eps=0.01,
        buffer_distance=0.05
    )
    
    return results

def main():
    """Run comprehensive geospatial analysis"""
    print("GEOSPATIAL ANALYSIS TOOL")
    print("=" * 40)
    
    data_dir = get_data_dir()
    output_dir = get_output_dir()
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Tourism hotspot analysis
    print("\nTOURISM ANALYSIS")
    print("=" * 60)
    tourism_keywords = ['zoo', 'wildlife', 'sanctuary', 'mountain', 'tourism', 'tourist']
    tourism_files = find_files_by_keywords(data_dir, tourism_keywords)
    
    if not tourism_files:
        tourism_files = list_spatial_files(data_dir)[:3]
    
    if tourism_files:
        analyzer = TourismHotspotAnalyzer()
        tourism_output = os.path.join(output_dir, 'tourism_analysis')
        tourism_results = analyzer.run_analysis(
            file_paths=tourism_files,
            data_dir=data_dir,
            output_dir=tourism_output,
            n_clusters=30
        )
        if tourism_results:
            print(f"Tourism: {tourism_results['total_points']} points in {tourism_results['clusters_created']} clusters")
    
    # Service gap analysis
    service_results = run_service_gap_analysis()
    if service_results:
        print(f"Service Coverage: {service_results['coverage_percentage']:.1f}%")
    
    print(f"\nOutputs saved to: {output_dir}")

if __name__ == "__main__":
    main()