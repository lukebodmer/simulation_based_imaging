# Inverse Model Training Features

New features for quantitative studies on inverse model performance.

## Signal Preprocessing

### Noise Injection
- **Parameter**: Noise level (0-100%)
- **Purpose**: Test model robustness to measurement noise
- **Study use**: Generate SNR vs reconstruction accuracy curves

### K-Space Mode
- **Toggle**: Use K-Space instead of raw voxel grid as model output
- **Purpose**: Train models in frequency domain
- **Study use**: Compare reconstruction quality between spatial and frequency domain approaches

## Sensor Selection

### Individual Sensor Control
- Grid-based UI matching the unfolded cube visualization
- Toggle individual sensors on/off per face
- "Sync across faces" option for symmetric configurations
- **Study use**:
  - Sensor sparsity studies (how few sensors needed?)
  - Optimal sensor placement analysis
  - Compare face-specific vs uniform coverage

## Evaluation Metrics

### Average Voxel Error
- Per-voxel MSE across test set
- **Study use**: Spatial error distribution analysis, identify problematic reconstruction regions

### K-Fold Cross Validation
- Configurable number of folds
- Per-fold model storage and evaluation
- Fold selection in Test tab for detailed analysis
- **Study use**: Robust performance estimates, variance analysis

## Suggested Experiments

1. **Noise robustness**: Train at 0%, 5%, 10%, 20% noise; compare test accuracy
2. **Sensor sparsity**: Progressively disable sensors; find minimum viable configuration
3. **K-space vs spatial**: Compare MLP/CNN performance in both domains
4. **Cross-validation**: Use k-fold to report mean ± std metrics
