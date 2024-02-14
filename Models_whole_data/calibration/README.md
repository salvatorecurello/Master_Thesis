# Calibration study 

A calibration study was conducted on the top two performing models:
- LegalLSGBERT+BiGRU which achieved an F1-score of 82.13, using the [CLS] token.
- CaseLawBERT+BiGRU, instead, achieved an F1-score of 81.12 also utilizing using the [CLS] token.

# File Descriptions

- calibration.ipynb: illustrates the metods used to analize the calibration according different metrics includec the rielability curve which is considered as the most important one. 
- temperature_scaling_calibration_lsgbert2560.ipynb: a calibration method called "Temperature scaling" is applied to LSGBERT (Non-Hierarchical) since its calibration curve was poorly aligned to the ideal one.
- /images: contains the images of the rielability curves and the histograms of the prediction