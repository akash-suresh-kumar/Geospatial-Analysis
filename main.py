#!/usr/bin/env python3
"""
Comprehensive Geospatial Analysis Tool
Main entry point for tourism hotspot and service gap analysis
"""

import os
from tourism_hotspot import TourismHotspotAnalyzer
from service_gap_analyzer import ServiceGapAnalyzer
from utils import (get_data_dir, get_output_dir, find_files_by_keywords, 
                   list_spatial_files, create_data_summary, test_clustering_parameters)

def run_tourism_analysis():
    """Run tourism hotspot analysis"""
    print("\n" + "=" * 60)
    print("TOURISM HOTSPOT ANALYSIS")
    print("=" * 60)
    
    data_dir = get_data_dir()
    output_dir = os.path.join(get_output_dir(), 'tourism_analysis')
    
    # Find tourism-related files
    tourism_keywords = ['zoo', 'wildlife', 'sanctuary', 'mountain', 'pass', 'tourism', 'tourist']
    tourism_files = find_files_by_keywords(data_dir, tourism_keywords)
    
    if not tourism_files:
        print("No tourism files found! Using all available files...")
        tourism_files = list_spatial_files(data_dir)[:3]
    
    if not tourism_files:
        print("No spatial files found for tourism analysis!")
        return None
    
    print(f"Analyzing files: {tourism_files}")
    
    analyzer = TourismHotspotAnalyzer()
    results = analyzer.run_analysis(
        file_paths=tourism_files,
        data_dir=data_dir,
        output_dir=output_dir,
        eps=0.1,
        min_samples=3
    )
    
    return results

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


def run_parameter_optimization():
    """Run parameter optimization for clustering"""
    print("\n" + "=" * 60)
    print("PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    data_dir = get_data_dir()
    output_dir = os.path.join(get_output_dir(), 'parameter_tests')
    
    all_files = list_spatial_files(data_dir)
    test_files = all_files[:3] if len(all_files) >= 3 else all_files
    
    if not test_files:
        print("No files available for parameter testing!")
        return None
    
    analyzer = TourismHotspotAnalyzer()
    best_params = test_clustering_parameters(analyzer, test_files, data_dir, output_dir)
    
    return best_params

def main():
    """Main function to run comprehensive geospatial analysis"""
    print("🌍 COMPREHENSIVE GEOSPATIAL ANALYSIS TOOL")
    print("=" * 60)
    
    # Setup directories
    data_dir = get_data_dir()
    output_dir = get_output_dir()
    
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\n❌ Data directory not found: {data_dir}")
        print("Please ensure your data files are in the 'data' folder.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data summary
    create_data_summary(data_dir)
    
    # Run analyses
    results = {}
    
    try:
        # Tourism hotspot analysis
        tourism_results = run_tourism_analysis()
        if tourism_results:
            results['tourism'] = tourism_results
        
        # Service gap analysis
        service_results = run_service_gap_analysis()
        if service_results:
            results['service_gap'] = service_results
        
        # Parameter optimization
        best_params = run_parameter_optimization()
        if best_params:
            results['best_params'] = best_params
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 60)
    
    if results:
        print("\n📊 RESULTS SUMMARY:")
        for analysis_type, result in results.items():
            if isinstance(result, dict) and 'total_points' in result:
                print(f"  {analysis_type.upper()}:")
                print(f"    Total points: {result['total_points']}")
                print(f"    Clustered points: {result['clustered_points']}")
                print(f"    Hotspots created: {result.get('hotspot_polygons', 'N/A')}")
            elif isinstance(result, dict) and 'coverage_percentage' in result:
                print(f"  {analysis_type.upper()}:")
                print(f"    Coverage: {result['coverage_percentage']:.1f}%")
                print(f"    Underserved areas: {result['underserved_areas_count']}")
    
    print(f"\n📁 All outputs saved to: {output_dir}")
    print("\n📋 Generated files:")
    print("  - tourism_heatmap.tiff (heatmap visualization)")
    print("  - tourism_hotspots.geojson (hotspot polygons)")
    print("  - clustered_points.geojson (clustered points)")
    print("  - well_served_areas.geojson (service coverage)")
    print("  - underserved_areas.geojson (service gaps)")
    print("  - poi_clusters.geojson (POI clusters)")

if __name__ == "__main__":
    main()