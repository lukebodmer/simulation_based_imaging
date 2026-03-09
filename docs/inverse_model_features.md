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

## Hyperparameter Sweep Study

Systematic optimization of neural network architectures for the inverse problem. Each sweep fixes all parameters except one, trains multiple models, and produces comparison charts for quantitative analysis.

### Methodology

1. Fix preprocessing: trim=45, downsample=4, voxel_grid=32, no k-space
2. Sweep one parameter at a time, measuring voxel error and loss
3. Select best value, then sweep next parameter
4. Produces charts (PNG/PDF) suitable for dissertation figures

### Scripts

Located in `packages/sbimaging/src/sbimaging/scripts/`:

```bash
python -m sbimaging.scripts.sweep_cnn2d_conv_channels --batch <batch_name>
```

### 2D CNN Sweep Results

#### Conv Channels (sweep_cnn2d_conv_channels)

| Conv Channels | Train Loss | Test Loss | Voxel Error |
|---------------|------------|-----------|-------------|
| [32, 64] | 2.26e-02 | 7.73e-02 | 3157.5 |
| **[64, 128]** | 2.21e-02 | **7.62e-02** | **2918.9** |
| [32, 64, 128] | 3.15e-02 | 7.89e-02 | 3125.3 |
| [64, 128, 256] | 2.51e-02 | 8.09e-02 | 3180.1 |
| [32, 64, 128, 256] | 3.05e-02 | 7.96e-02 | 3094.8 |

**Best**: `[64, 128]` — lowest voxel error (2918.9) and test loss (7.62e-02)

**Observation**: Deeper networks (3-4 conv layers) don't improve performance and show signs of overfitting (higher test loss despite similar train loss). The simpler 2-layer architecture generalizes better.

#### Pool Size (sweep_cnn2d_pool_size)

Fixed: `conv_channels=[64, 128]`

| Pool Size | Train Loss | Test Loss | Voxel Error |
|-----------|------------|-----------|-------------|
| 8x8 | 2.26e-02 | 7.84e-02 | 3341.1 |
| 12x12 | 2.31e-02 | 7.68e-02 | 3075.5 |
| 12x16 | 2.22e-02 | 7.67e-02 | 2886.3 |
| 16x16 | 2.24e-02 | 7.63e-02 | 2872.5 |
| **12x24** | 2.24e-02 | **7.63e-02** | **2827.1** |

**Best**: `(12, 24)` — lowest voxel error (2827.1)

**Observation**: Preserving more temporal information (wider pool) improves reconstruction. The trend suggests temporal resolution is more valuable than spatial compression at this stage. The 12x24 configuration preserves sensor spatial structure (12) while retaining more timestep information (24).

#### Regressor Hidden (sweep_cnn2d_regressor_hidden)

Fixed: `conv_channels=[64, 128]`, `pool_size=(12, 24)`

| Regressor Hidden | Train Loss | Test Loss | Voxel Error |
|------------------|------------|-----------|-------------|
| **512** | 3.54e-02 | **7.48e-02** | **2662.3** |
| 1024 | 2.24e-02 | 7.61e-02 | 2785.3 |
| 2048 | 1.28e-02 | 7.78e-02 | 3079.5 |
| 4096 | 7.18e-03 | 7.79e-02 | 3087.9 |

**Best**: `512` — lowest voxel error (2662.3) and test loss (7.48e-02)

**Observation**: Classic overfitting pattern. Larger hidden layers drive train loss down dramatically (7.18e-03 vs 3.54e-02) but test loss and voxel error increase. The smaller 512-unit regressor generalizes best, suggesting the bottleneck acts as regularization. With pool output of 128×12×24=36,864 features, a compact regressor forces the network to learn efficient representations.

#### Dropout (sweep_cnn2d_dropout)

Fixed: `conv_channels=[64, 128]`, `pool_size=(12, 24)`, `regressor_hidden=512`

| Dropout | Train Loss | Test Loss | Voxel Error |
|---------|------------|-----------|-------------|
| 0.1 | 1.80e-02 | 7.69e-02 | 3126.5 |
| 0.2 | 2.77e-02 | 7.58e-02 | 2922.6 |
| 0.3 | 3.51e-02 | 7.47e-02 | 2686.5 |
| 0.4 | 4.42e-02 | 7.48e-02 | 2483.8 |
| **0.5** | 5.29e-02 | **7.48e-02** | **2391.6** |

**Best**: `0.5` — lowest voxel error (2391.6)

**Observation**: Strong regularization helps significantly. Higher dropout consistently improves voxel error despite increasing train loss. The trend suggests the model benefits from aggressive regularization — the compact 512-unit regressor combined with 50% dropout prevents overfitting to the training set. Test loss plateaus around 0.3-0.5, but voxel error continues to improve.

#### Kernel Size (sweep_cnn2d_kernel_size)

Fixed: `conv_channels=[64, 128]`, `pool_size=(12, 24)`, `regressor_hidden=512`, `dropout=0.5`

| Kernel Size | Train Loss | Test Loss | Voxel Error |
|-------------|------------|-----------|-------------|
| 1x11 | 5.48e-02 | **7.31e-02** | 2562.1 |
| **(3, 5)** | 5.24e-02 | 7.46e-02 | **2427.3** |
| 3x7 | 5.54e-02 | 7.62e-02 | 2620.4 |
| 3x11 | 5.64e-02 | 7.58e-02 | 2674.5 |
| 3x15 | 5.58e-02 | 7.60e-02 | 2657.8 |
| 5x11 | 5.60e-02 | 7.58e-02 | 2605.2 |
| 7x7 | 5.65e-02 | 7.67e-02 | 2622.0 |
| 7x11 | 5.39e-02 | 7.63e-02 | 2438.5 |

**Best**: `(3, 5)` — lowest voxel error (2427.3)

**Observation**: Smaller temporal kernel works best. The (3, 5) configuration outperforms the original (3, 11) default significantly. This suggests that with downsampling factor=4, the effective temporal resolution is already compressed, so a smaller kernel captures the relevant wave features without oversmoothing. Interestingly, (1, 11) achieves the lowest test loss but not the best voxel error, indicating these metrics can diverge.

#### Stride (sweep_cnn2d_stride)

Fixed: `conv_channels=[64, 128]`, `pool_size=(12, 24)`, `regressor_hidden=512`, `dropout=0.5`, `kernel_size=(3, 5)`

| Stride | Train Loss | Test Loss | Voxel Error |
|--------|------------|-----------|-------------|
| 1x1 | 5.37e-02 | 7.46e-02 | 2531.4 |
| 1x2 | 5.31e-02 | 7.46e-02 | 2457.9 |
| 2x2 | 5.74e-02 | 7.44e-02 | 2745.5 |
| **(1, 3)** | 5.33e-02 | **7.45e-02** | **2373.9** |

**Best**: `(1, 3)` — lowest voxel error (2373.9)

**Observation**: Aggressive temporal downsampling helps. The (1, 3) stride preserves full sensor spatial resolution while compressing time by 3x per conv layer. This aligns with earlier findings — temporal compression is beneficial (also seen in pool_size results). The (2, 2) uniform stride performs worst, confirming that preserving sensor spatial relationships is important.

#### Use Residual (sweep_cnn2d_use_residual)

Fixed: `conv_channels=[64, 128]`, `pool_size=(12, 24)`, `regressor_hidden=512`, `dropout=0.5`, `kernel_size=(3, 5)`, `stride=(1, 3)`

| Use Residual | Train Loss | Test Loss | Voxel Error |
|--------------|------------|-----------|-------------|
| False | 5.43e-02 | **7.06e-02** | 2606.3 |
| **True** | 5.48e-02 | 7.49e-02 | **2534.4** |

**Best**: `True` — lowest voxel error (2534.4)

**Observation**: Residual connections improve voxel error but increase test loss. This is an interesting divergence — the model without residuals achieves better MSE loss but worse voxel reconstruction. This suggests residual connections help preserve structural information important for accurate voxel reconstruction, even if the raw numerical loss is slightly higher.

### Optimized 2D CNN Configuration

Based on the sweep study, the optimal 2D CNN configuration is:

| Parameter | Value |
|-----------|-------|
| conv_channels | [64, 128] |
| pool_size | (12, 24) |
| regressor_hidden | 512 |
| dropout | 0.5 |
| kernel_size | (3, 5) |
| stride | (1, 3) |
| use_residual | True |

**Final voxel error**: ~2400-2500 (improved from initial ~3100)

### MLP Sweep Results

#### Hidden Layers (sweep_mlp_hidden_layers)

| Hidden Layers | Train Loss | Test Loss | Voxel Error |
|---------------|------------|-----------|-------------|
| [2048, 1024] | 3.11e-02 | 7.48e-02 | 3223.5 |
| [4096, 2048] | 2.28e-02 | 7.33e-02 | 3287.1 |
| [4096, 2048, 1024] | 3.31e-02 | 7.58e-02 | 3048.9 |
| [4096, 2048, 1024, 512] | 4.58e-02 | 8.06e-02 | 2834.2 |
| [8192, 4096, 2048] | 2.27e-02 | 7.46e-02 | 2925.5 |
| **[8192, 4096, 2048, 1024]** | 2.54e-02 | 7.76e-02 | **2822.4** |

**Best**: `[8192, 4096, 2048, 1024]` — lowest voxel error (2822.4)

**Observation**: Deeper and wider networks perform better for MLP. The 4-layer configuration with large initial layers (8192) achieves the best voxel error. Unlike the 2D CNN where compact architectures generalized better, the MLP benefits from more parameters — likely because it lacks the inductive bias of convolutions and needs raw capacity to learn the mapping.

#### Dropout (sweep_mlp_dropout)

Fixed: `hidden_layers=[8192, 4096, 2048, 1024]`

| Dropout | Train Loss | Test Loss | Voxel Error |
|---------|------------|-----------|-------------|
| 0.1 | 2.67e-02 | 7.75e-02 | 2832.7 |
| 0.2 | 2.41e-02 | 7.82e-02 | 2810.6 |
| 0.3 | 3.20e-02 | 7.87e-02 | 2911.2 |
| 0.4 | 2.95e-02 | 7.94e-02 | 2761.0 |
| **0.5** | 2.61e-02 | 8.05e-02 | **2581.4** |

**Best**: `0.5` — lowest voxel error (2581.4)

**Observation**: Similar to the 2D CNN, higher dropout improves generalization. The MLP shows a clear trend where voxel error decreases with increased dropout despite test loss increasing. At 50% dropout, voxel error improves by ~9% compared to 10% dropout (2581.4 vs 2832.7). The large capacity of the [8192, 4096, 2048, 1024] architecture benefits from aggressive regularization.

### 1D CNN Sweep Results

#### Conv Channels (sweep_cnn1d_conv_channels)

| Conv Channels | Train Loss | Test Loss | Voxel Error |
|---------------|------------|-----------|-------------|
| [32, 64] | 3.55e-02 | 8.86e-02 | 3835.2 |
| [64, 128] | 3.26e-02 | 8.75e-02 | 3531.6 |
| [32, 64, 128] | 3.53e-02 | 8.20e-02 | 3518.0 |
| **[64, 128, 256]** | 3.47e-02 | **8.15e-02** | **3350.5** |
| [32, 64, 128, 256] | 3.29e-02 | 7.79e-02 | 3417.4 |

**Best**: `[64, 128, 256]` — lowest voxel error (3350.5)

**Observation**: The 1D CNN performs worse than both the 2D CNN (~2400) and MLP (~2580) on this task. The best configuration uses 3 layers with [64, 128, 256] channels. Unlike the 2D CNN where 2 layers worked best, the 1D CNN benefits from deeper architectures. This may be because the 1D CNN treats the flattened sensor data as a single sequence, losing the 2D spatial structure that the 2D CNN exploits.

#### Pool Size (sweep_cnn1d_pool_size)

Fixed: `conv_channels=[64, 128, 256]`

| Pool Size | Train Loss | Test Loss | Voxel Error |
|-----------|------------|-----------|-------------|
| 8 | 3.38e-02 | 8.31e-02 | 3619.1 |
| 16 | 3.25e-02 | 8.05e-02 | 3402.7 |
| 24 | 3.16e-02 | 8.05e-02 | 3387.1 |
| 32 | 2.49e-02 | 8.19e-02 | 3359.4 |
| **48** | 2.68e-02 | **7.96e-02** | **3223.8** |

**Best**: `48` — lowest voxel error (3223.8)

**Observation**: Larger pool sizes consistently improve performance for 1D CNN. The trend is monotonic — more pooled features lead to better reconstruction. This differs from the 2D CNN where an intermediate pool size was optimal. The 1D CNN may need more features to compensate for its lack of 2D spatial awareness.

#### Regressor Hidden (sweep_cnn1d_regressor_hidden)

Fixed: `conv_channels=[64, 128, 256]`, `pool_size=48`

| Regressor Hidden | Train Loss | Test Loss | Voxel Error |
|------------------|------------|-----------|-------------|
| **256** | 5.27e-02 | **7.70e-02** | **2770.2** |
| 512 | 3.70e-02 | 7.87e-02 | 2917.2 |
| 1024 | 3.02e-02 | 8.03e-02 | 3275.7 |
| 2048 | 2.39e-02 | 8.13e-02 | 3531.6 |
| 4096 | 1.94e-02 | 8.28e-02 | 3647.3 |

**Best**: `256` — lowest voxel error (2770.2)

**Observation**: Classic overfitting pattern, even more pronounced than the 2D CNN. Larger regressors dramatically reduce train loss (1.94e-02 vs 5.27e-02) but increase test loss and voxel error. The smallest 256-unit regressor generalizes best. With pool output of 256×48=12,288 features, a compact regressor acts as a strong regularizer. This is consistent with the 2D CNN finding where 512 worked best.

#### Dropout (sweep_cnn1d_dropout)

Fixed: `conv_channels=[64, 128, 256]`, `pool_size=48`, `regressor_hidden=256`

| Dropout | Train Loss | Test Loss | Voxel Error |
|---------|------------|-----------|-------------|
| 0.1 | 3.57e-02 | 7.74e-02 | 3067.4 |
| 0.2 | 4.37e-02 | 7.66e-02 | 2888.3 |
| 0.3 | 5.24e-02 | 7.55e-02 | 2806.3 |
| **0.4** | 6.02e-02 | **7.60e-02** | **2525.6** |
| 0.5 | 7.12e-02 | 8.01e-02 | 2966.0 |

**Best**: `0.4` — lowest voxel error (2525.6)

**Observation**: Dropout of 0.4 provides the best balance. Unlike the 2D CNN and MLP where 0.5 was optimal, the 1D CNN with its already-compact 256-unit regressor benefits from slightly less aggressive dropout. At 0.5, performance degrades — the combination of small regressor and high dropout may be too restrictive, causing underfitting.

### Optimized 1D CNN Configuration

Based on the sweep study, the optimal 1D CNN configuration is:

| Parameter | Value |
|-----------|-------|
| conv_channels | [64, 128, 256] |
| pool_size | 48 |
| regressor_hidden | 256 |
| dropout | 0.4 |
| use_residual | True |

**Final voxel error**: ~2525 (improved from initial ~3350)

## Noise Robustness Study

To evaluate model robustness to measurement noise, we conducted a 5-fold cross-validation study at different noise levels using the optimized 2D CNN configuration. Gaussian noise was added to the sensor data as a percentage of the global signal peak.

### Methodology

1. Use optimized 2D CNN parameters from hyperparameter sweep
2. Add Gaussian noise at 0%, 5%, and 10% of global peak amplitude
3. Train 5-fold cross-validation for each noise level
4. Report mean ± std for train loss, test loss, and voxel error

### Results (study_noise_robustness)

| Noise Level | Train Loss | Test Loss | Voxel Error |
|-------------|------------|-----------|-------------|
| **0%** | 5.03e-02 ± 1.01e-03 | **8.33e-02 ± 3.17e-03** | **2468.8 ± 49.6** |
| 5% | 5.21e-02 ± 1.70e-03 | 9.44e-02 ± 2.96e-03 | 2783.2 ± 113.7 |
| 10% | 5.25e-02 ± 1.92e-03 | 1.07e-01 ± 4.54e-03 | 2772.8 ± 75.6 |

**Observation**: The model shows graceful degradation with noise. At 5% noise, voxel error increases by ~13% (2469 → 2783). Interestingly, 10% noise shows similar voxel error to 5% (2773 vs 2783), despite higher test loss. This suggests the model learns to be somewhat robust to noise beyond a threshold. The low standard deviations across folds indicate stable performance. Train loss remains relatively constant across noise levels, while test loss increases proportionally with noise — the model fits the noisy training data but generalizes worse on clean test data patterns.
