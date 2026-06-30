# Rohan's Contribution

## Purpose
Implemented and evaluated multiple approaches for extracting engineering information from highway cross-section PDFs, removing annotations, and preserving engineering geometry. Also conducted feasibility studies on different PDF types and analyzed image quality after annotation removal.

---

# Approach 1 – Image Processing Based Workflow

### Objective
To process scanned/raster engineering drawings using image processing and OCR techniques.

### Work Completed
- Implemented 19-series page extraction.
- Identified Cross Section drawings.
- Extracted Horizontal and Vertical scales.
- Removed graph labels and numerical annotations.
- Evaluated OCR-based annotation removal using EasyOCR.
- Developed rule-based annotation removal techniques.
- Developed aggressive morphology-based annotation removal.
- Removed engineering grid lines.
- Analyzed output quality and preservation of engineering geometry.
- Conducted pixel and granularity analysis on SAM 2/3 outputs.

### Modules
- extract_19_series_pages.py
- number_removal.py
- easyocr_annotation_removal.py
- aggressive_morphology_annotation_removal.py
- rule_based_annotation_removal.py
- grid_removal.py

---

# Approach 2 – Vector PDF Processing

### Objective
To process vector/embedded PDFs directly without OCR by extracting engineering information from PDF objects.

### Work Completed
- Identified vector and embedded PDFs.
- Studied PDF object structure.
- Removed annotations directly from vector PDFs.
- Evaluated feasibility of extracting engineering elements without rasterization.
- Compared vector PDF processing with image-based processing.

### Modules
- vector_pdf_annotation_removal.py

---

# Reports

The Reports folder contains:

- PDF Processing Feasibility Study
- SAM 2/3 Pixel and Granularity Analysis
- Image Quality Analysis
- PDF Type Identification Report

---

# Completed Work

- PDF classification
- Raster PDF analysis
- Embedded PDF analysis
- Vector PDF analysis
- 19-series page extraction
- Cross Section identification
- Scale extraction
- Number removal
- OCR-based annotation removal
- Rule-based annotation removal
- Morphological annotation removal
- Vector PDF annotation removal
- Grid removal
- Pixel and granularity analysis
- Feasibility study for PDF processing

---

# Current Work

- Improving annotation removal accuracy
- Preserving engineering geometry
- Performance optimization
- Validation of extracted engineering information
- Evaluating vector PDF extraction techniques

---

# Dependencies

Python 3.10+

Libraries Used

- PyMuPDF (fitz)
- OpenCV
- NumPy
- EasyOCR
- pytesseract
- Pillow
- matplotlib
- re
- os
- shutil
- zipfile                                                    



# Approach 1 – Traditional Computer Vision Based Earthwork Calculation

## Overview

This approach focused on investigating the feasibility of using traditional computer vision techniques to automate earthwork quantity estimation from highway cross-section drawings.

The study explored how engineering profiles could be isolated from complex drawings containing annotations, bridge geometry, utility symbols, leader lines, arrows, and other graphical elements.

The primary objective was to determine whether Existing Ground (EG) and Proposed Grade (PG) profiles could be reliably extracted for downstream engineering calculations.

---

## Stage 1 – Image Preparation

### Objective
Evaluate techniques for preparing engineering drawings for further analysis.

### Activities Performed
- Investigated background grid removal techniques.
- Evaluated methods to preserve engineering profile geometry.
- Studied preprocessing techniques for improving image quality.
- Compared image quality before and after preprocessing.

### Outcome
- Successfully generated cleaner cross-section images for further analysis.
- Identified preprocessing methods that preserved profile geometry while reducing background noise.

---

## Stage 2 – Existing Ground (EG) and Proposed Grade (PG) Identification

### Objective
Investigate methods for identifying the two primary engineering profiles.

### Activities Performed
- Studied Connected Component Analysis for engineering drawings.
- Evaluated geometric properties such as:
  - Bounding Box
  - Area
  - Width
  - Height
  - Aspect Ratio
- Investigated rule-based techniques for distinguishing engineering profiles from annotations.

### Outcome
- Identified candidate profile components.
- Observed limitations caused by overlapping annotations and engineering objects.

---

## Stage 3 – Curve Extraction

### Objective
Evaluate methods for converting engineering profiles into coordinate data.

### Activities Performed
- Investigated column-wise profile tracing.
- Studied interpolation methods for missing profile segments.
- Evaluated profile continuity after preprocessing.

### Outcome
- Developed an understanding of profile tracing techniques.
- Identified challenges in maintaining continuous profile geometry when annotations intersected engineering lines.

---

## Stage 4 – Cut and Fill Area Analysis

### Objective
Study numerical methods for calculating cut and fill areas.

### Activities Performed
- Investigated comparison of Existing Ground and Proposed Grade coordinates.
- Evaluated numerical integration techniques.
- Studied engineering methodologies used for cut/fill calculations.

### Outcome
- Understood the workflow required for area calculation.
- Identified prerequisite accuracy requirements for reliable profile extraction.

---

## Stage 5 – Volume Calculation

### Objective
Study engineering methods used for earthwork volume estimation.

### Activities Performed
- Investigated the Average End Area Method.
- Reviewed engineering workflows for volume computation between adjacent stations.

### Outcome
- Documented the engineering calculation workflow.
- Determined that accurate profile extraction is critical before reliable volume calculations can be performed.

---

## Key Findings

During the feasibility study, several practical challenges were identified:

- Engineering annotations frequently overlap profile lines.
- Bridge geometry introduces additional complexity.
- OCR alone is insufficient for reliable profile extraction.
- Rule-based image processing improves results but has limitations.
- Profile continuity is difficult to maintain in heavily annotated drawings.

These findings motivated further work on annotation removal and PDF processing approaches.

---

## Status

**Current Status:** Feasibility Study Completed

The investigation provided a clear understanding of the challenges involved in traditional computer vision–based earthwork calculation and served as the foundation for subsequent work on annotation removal, vector PDF processing, and PDF classification.