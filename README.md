# Grammar-Scoring-Engine

# Grammar Scoring Engine with Audio Features & Model Stacking

**Project Overview**  
By extracting acoustic features and leveraging a stacked ensemble model, I achieved a **Pearson correlation of 0.9319** between predicted and human-labeled scores, demonstrating strong generalization.

## Model Performance Visualization

### Base Model Comparisons
| Metric          | Top Performers               |
|-----------------|------------------------------|
| **Lowest RMSE** | SVR → ET → XGB               |
| **Highest r**   | SVR (0.888) → ET (0.887)     |

![Cross-Validation RMSE by Model](results/output.png)  
*Base models' RMSE scores (lower is better)*  

![Pearson Correlation by Model](results/output%20(1).png)  
*Pearson r values (closer to 1 indicates stronger linear relationship)*  

![Model Performance Trade-off](results/output%20(2).png)  
**Key Insight**: SVR/ET achieve the best balance of low RMSE and high correlation 

## Features
- **Audio Preprocessing**:  
  All audio files are resampled to 16 kHz and augmented with pitch shifts (±2 steps) and time stretches (0.9x, 1.1x) to improve robustness.

- **Feature Extraction (98-dimensional)**:  
  Extracted using Librosa:
  - **MFCCs**: 20 coefficients (mean + std)
  - **Chroma**: 12 pitch classes (mean + std)
  - **Spectral Contrast**: 7 frequency bands (mean + std)
  - **Tonnetz**: 6 tonal features (mean + std)
  - **Temporal Features**: Beat tempo, RMS energy, Zero-Crossing Rate, and audio statistics.


## Model Architecture
A two-stage stacked ensemble approach:

### **Base Models**
- Gradient Boosting (GBR)
- Random Forest (RF)
- Extra Trees (ET)
- Support Vector Regressor (SVR)
- K-Nearest Neighbors (KNN)
- AdaBoost (Ada)
- Histogram Gradient Boosting (HistGB)
- XGBoost (XGB)

### **Meta Model**
- **SVR** trained on out-of-fold predictions from base models using **5-fold cross-validation**.

## Performance 

**Stacked Model Performance**  
- **Train RMSE**: 0.3805  
- **Pearson r**: 0.9319  


### Workflow
1. **Feature Extraction**:  
   Run `build_augmented_data()` to generate augmented training features and labels.
2. **Model Training**:  
   Execute `stacking()` to train base models and meta-learner.

- **Best Base Model**: SVR achieved the lowest RMSE (0.4839) and highest Pearson r (0.8882).
- **Stacking Benefit**: The meta-learner (SVR) improved performance by **4.8%** in Pearson r over the best base model.
- **Augmentation Impact**: Training on pitch/time-augmented data reduced overfitting.

