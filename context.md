1. Project Overview

An advanced ensemble supervised learning system for Land Use and Land Cover (LULC) classification in Arunachal Pradesh. The project integrates multi-sensor data (Sentinel-2, Landsat), high-dimensional embeddings (Google Satellite Embedding V1, Gemini Embedding 2), and topographic corrections to handle the Eastern Himalayas' unique challenges.
2. Data Sources & Acquisition

    Radiometric Data: Sentinel-2 (10-20m) and Landsat 8/9 (30m).

    Foundation Embeddings: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL (AEF V1). 64-D vectors summarizing annual trajectories.

    Multimodal Fusion: Gemini Embedding 2. A 3072-D unified space for text, images, and tabular data.

    Topography: ALOS World 3D or SRTM (30m) DEMs for slope, aspect, and hillshade.

3. Preprocessing & Cloud Reduction

Arunachal Pradesh suffers from chronic cloud cover. The following algorithms are implemented for testing and production:
3.1 Base Mosaicing Algorithm (User-Provided)

The system currently uses a qualityMosaic based on the inverse of the CLOUD_COVER metadata property and a QA_PIXEL bitmask --> In the old_code/cloud_masking_algo.js file


3.2 Advanced Cloud-Removal Alternatives for Testing

    Cloud Score Plus (CS+): Use GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED. This is a machine-learning-based QA score that is significantly more accurate than standard bitmasks for Sentinel-2.

    s2cloudless: Use COPERNICUS/S2_CLOUD_PROBABILITY. This provides a pixel-wise probability (0–100) instead of a binary mask, allowing for finer control in "borderline" cloud areas.

    Temporal Dark-Outlier Mask (TDOM): Specifically designed to identify and mask cloud shadows, which are often missed by standard QA bands in mountainous terrain.

    Medoid Mosaicing: Instead of a simple median or quality mosaic, use a Medoid composite. The medoid selects the actual observed pixel that is spectrally closest to the median of the entire stack, preserving the original spectral relationship between bands.

4. Mathematical Foundations for Robust Testing

To prevent data leakage and ensure the model generalizes across the massif, implement:

    Moran's I / Geary’s C: Calculate these statistics on your NDVI/Slope layers to determine the distance threshold (phi) where points are no longer spatially dependent.

    kNNDM (k-fold Nearest Neighbor Distance Matching): Generate validation folds that match the distance distribution between your training set and the entire prediction area. This ensures that the "difficulty" of your test set matches the real-world prediction task.

    Adversarial Validation (DAV): Use a binary classifier to attempt to distinguish between training and test sets. A high classification accuracy here indicates the test set is "different" (out-of-distribution), which is ideal for a robustness check.

    Feature Space Dissimilarity: Ensure test points cover different environmental "envelopes" (e.g., ensure you have test points for high-altitude bamboo if your training data is primarily mid-altitude).

5. Model Architecture: Synergistic Neural Ensemble

    Spatial Extractor: CNN (U-Net/ResNet) for patch-based texture features.

    Contextual Module: Vision Transformer (ViT) for global dependencies.

    Traditional Head: SVM or Random Forest for tabular/topographic data.

    Meta-Learner: An MLP stacking these outputs into a final class prediction.

6. Implementation Roadmap: Step-by-Step
Stage 1: Cloud-Reduction A/B Testing

    Test C1 (Control): User-provided qualityMosaic script.

    Test C2 (Experimental): Medoid Mosaic + s2cloudless.

    Test C3 (Experimental): Cloud Score Plus (Sentinel-2 only).

    Metric: Visually inspect for artifacts in steep shadows and calculate the standard deviation of NDVI in "stable" forest pixels (lower SD indicates better cloud/shadow removal).

Stage 2: Topographic Correction

    Apply SCS+C correction to the best mosaic from Stage 1 using the ALOS DEM.

Stage 3: Feature Engineering & Multimodal Fusion

    Fetch AEF V1 embeddings.

    Pass false-color composites + tabular indices (NDVI, NDWI) to Gemini Embedding 2 to retrieve the 3072-D unified feature vectors.

Stage 4: Robust Training/Validation Setup

    Run Moran's I to find the spatial autocorrelation range.

    Partition data into folds using kNNDM.

    Verify test points are "different" enough via Adversarial Validation.

Stage 5: Imbalance Handling & Ensemble Training

    Apply K-means SMOTE to Bamboo and Grassland (merging Synonymous labels first).

    Train the CNN/ViT and RF heads.

    Train the Stacking Meta-Learner on the hold-out validation set.

Stage 6: Temporal Verification

    Run the BFAST pipeline on the final results to verify Shifting Cultivation cycles.

    Use "Stable Points" (no change in 5+ years) to refine the final land-cover boundaries.

7. Evaluation Metrics

Prioritize F1-Score, mIoU, and Kappa for imbalanced categories (Bamboo, Grassland, Shifting Cultivation).