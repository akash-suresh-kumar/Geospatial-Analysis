
# Comprehensive Geospatial Analysis Tool

This repository provides a complete Python-based toolkit for performing two types of geospatial analysis:

1. **Tourism Hotspot Analysis**
2. **Service Gap Analysis**

It works with spatial vector datasets such as `.shp`, `.geojson`, `.gpkg`, `.json`, `.kml`, and `.gml`. The toolkit uses clustering algorithms (KMeans and DBSCAN), spatial buffers, and raster heatmaps to extract geographic patterns and service coverage insights.

---

## Features

### Tourism Hotspot Analysis
- Clusters tourism-related POIs using **KMeans**.
- Creates hotspot polygons via **buffer** and **convex hull**.
- Generates a **raster heatmap** using KDE.
- Outputs: 
  - `clustered_points.geojson`
  - `tourism_hotspots.geojson`
  - `tourism_heatmap.tiff`

### Service Gap Analysis
- Clusters service POIs (e.g., hospitals, schools) using **DBSCAN**.
- Buffers clusters to model service areas.
- Detects **underserved regions** within administrative boundaries.
- Outputs:
  - `poi_clusters.geojson`
  - `service_areas.geojson`
  - `well_served_areas.geojson`
  - `underserved_areas.geojson`

### Input Handling
- Scans `data/` directory for spatial files.
- Keyword-based auto detection for POIs and boundary layers.

### Modular Design
- Separated logic for each analysis type.
- Common utilities for file operations and CLI orchestration.

---

## Directory Structure

```
geospatial-analysis/
├── data/                           # Input spatial datasets
├── Outputs/                        # All result outputs (GeoJSONs, TIFFs)
│   ├── comprehensive_tourism_analysis/
│   │   ├── clustered_points.geojson
│   │   ├── tourism_hotspots.geojson
│   │   └── tourism_heatmap.tiff
│   └── comprehensive_service_gap_analysis/
│       ├── poi_clusters.geojson
│       ├── service_areas.geojson
│       ├── well_served_areas.geojson
│       └── underserved_areas.geojson
├── main.py                         # CLI entry point
├── tourism_hotspot.py             # TourismHotspotAnalyzer class
├── service_gap_analyzer.py        # ServiceGapAnalyzer class
├── utils.py                        # File handling utilities
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/akash-suresh-kumar/Geospatial-Analysis.git
cd geospatial-analysis
```

### 2. Set up virtual environment

```bash
python -m venv iudx
source ./iudx/Scripts/activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Output Structure

All output files are saved to the `Outputs/` directory under analysis-specific subfolders.

**Tourism Hotspot Outputs**
- `clustered_points.geojson`
- `tourism_hotspots.geojson`
- `tourism_heatmap.tiff`
- Location: `Outputs/comprehensive_tourism_analysis/`

**Service Gap Analysis Outputs**
- `poi_clusters.geojson`
- `service_areas.geojson`
- `well_served_areas.geojson`
- `underserved_areas.geojson`
- Location: `Outputs/comprehensive_service_gap_analysis/`