# Comprehensive Geospatial Analysis Tool

This repository provides a complete Python-based toolkit for performing two types of geospatial analysis:

1. **Tourism Hotspot Analysis**
2. **Service Gap Analysis**

It is designed to work with spatial vector datasets such as `.shp`, `.geojson`, `.gpkg`, and others. The toolkit leverages clustering algorithms (KMeans and DBSCAN), spatial buffers, and raster heatmaps to generate analytical outputs about geographic patterns and service coverage.

---

## Features

- **Tourism Hotspot Analysis**
  - Detects and clusters tourism-related POIs using KMeans.
  - Creates hotspot polygons with buffer and convex hull techniques.
  - Generates a raster heatmap using KDE (Kernel Density Estimation).
  - Outputs GeoJSONs and a `.tiff` heatmap raster.

- **Service Gap Analysis**
  - Clusters service POIs (e.g., hospitals, schools) using DBSCAN.
  - Buffers clustered POIs to create service areas.
  - Identifies underserved regions within defined administrative boundaries.
  - Computes statistics on total area, served area, and underserved count.

- **Input Management**
  - Automatically detects relevant files using keyword-based filters.
  - Supports `.shp`, `.geojson`, `.gpkg`, `.json`, `.kml`, and `.gml`.

- **Modular Architecture**
  - Clearly separated components for hotspot analysis, service coverage, utilities, and CLI orchestration.

---

## Directory Structure

geospatial-analysis/
├── data/ # Input spatial datasets
├── Outputs/ # All result outputs (GeoJSONs, TIFFs)
│
├── main.py # CLI entry point for users
├── tourism_hotspot.py # TourismHotspotAnalyzer class
├── service_gap_analyzer.py # ServiceGapAnalyzer class
├── utils.py # Utility functions for file handling
├── requirements.txt # Python package dependencies
└── README.md # Project documentation


---

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/akash-suresh-kumar/Geospatial-Analysis.git
cd geospatial-analysis

### Set up Virtual Environment
python -m venv iudx
source .\iudx\Scripts\activate

##Install Dependencies
pip install -r requirements.txt

###Output Details
All output files will be created in the Outputs/ directory, in subfolders named after the analysis type.

Tourism Hotspot Outputs
clustered_points.geojson

tourism_hotspots.geojson

tourism_heatmap.tiff

Location:

Outputs/comprehensive_tourism_analysis/
Outputs/comprehensive_tourism_analysis/detailed/
Service Gap Analysis Outputs
poi_clusters.geojson

service_areas.geojson

well_served_areas.geojson

underserved_areas.geojson

Location:

Outputs/comprehensive_service_gap_analysis/
Outputs/comprehensive_service_gap_analysis/detailed/


