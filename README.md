# Flood Mapping from Sentinel-2 Imagery (Brzeg, Poland)

[![CI](https://github.com/hcroysdill-dot/Flood_Prediction_Model/actions/workflows/ci.yml/badge.svg)](https://github.com/hcroysdill-dot/Flood_Prediction_Model/actions/workflows/ci.yml)

A Python pipeline that classifies flooded vs. not-flooded areas from Sentinel-2 RGB imagery using machine learning. It compares four scikit-learn models, trains the chosen model on labelled ground-truth points, and maps the predicted flood extent.

**Study area:** Brzeg, Poland, 18 September 2024
**Data:** Sentinel-2 imagery from the Copernicus Browser. QGIS was used for visualisation and for creating the ground-truth points.

## Workflow

1. **Prepare data:** clip and resample the image, then extract the RGB values at each labelled ground-truth point (1 = flooded, 0 = not flooded).
2. **Compare models (Execution 1):** train and test four classifiers on an 80/20 stratified split:
   - Random Forest
   - Logistic Regression
   - Support Vector Classifier (SVC)
   - Multi-Layer Perceptron (MLP)
3. **Predict (Execution 2):** train the selected model on all ground-truth points and classify the remaining pixels, which are represented as centroid points.
4. **Map:** convert flooded points to polygons and overlay them on the satellite image.

## Results (held-out test points)

| Model | F1, not flooded | F1, flooded |
|---|---|---|
| Random Forest | 0.87 | 0.88 |
| Logistic Regression | 0.86 | 0.86 |
| SVC | 0.83 | 0.84 |
| MLP | 0.89 | 0.88 |

Random Forest was used for the final map. Its scores were close to the MLP's, and the MLP gave inconsistent results between runs.

## Limitations

- The model uses RGB values only, with no infrared or water-index bands.
- Evaluation uses a small set of points from a single image and date, so results may not generalise to other events or locations.
- Independent validation against ground-truth polygons is planned.

## Tech stack

Python, scikit-learn, rasterio, GeoPandas, Shapely, NumPy, Matplotlib, QGIS, CI: GitHub Actions (pytest, ruff)

## Usage

1. Install dependencies: `pip install rasterio geopandas scikit-learn matplotlib`
2. Set the file paths in the "Variables" cells (the notebook was written for Google Colab with data on Google Drive).
3. Run the notebook top to bottom.
