# 3D-Printed Object Classification

Transfer learning comparison of **VGG16**, **ResNet50** and **DenseNet121** for classifying 3D-printed versus conventionally manufactured objects.

All three architectures are fine-tuned on an identical pipeline and evaluated on the same test set, so differences in performance reflect the backbone rather than the surrounding setup.

## Results

| Model | Accuracy | F1-score | Total parameters |
| --- | --- | --- | --- |
| **VGG16** | **0.9167** | **0.9186** | 14,846,273 |
| ResNet50 | 0.9033 | 0.9030 | 24,112,513 |
| DenseNet121 | 0.8833 | 0.8829 | 7,300,161 |

Evaluated on 300 test images (150 per class).

Two observations:

- **Depth does not determine size.** DenseNet121 has 121 weight layers to ResNet50's 50, yet under a third of the parameters — dense connectivity reuses features instead of relearning them.
- **Size does not determine performance.** DenseNet121 came within 3.3 percentage points of the best model using 7.3M parameters against ResNet50's 24.1M, suggesting the task is not limited by model capacity.

The margins between models are small relative to the test set size (95% CI ≈ ±3 percentage points), so the ranking is indicative rather than conclusive.

### VGG16 confusion matrix

|  | Predicted 0 | Predicted 1 |
| --- | --- | --- |
| **Actual 0** | 133 | 17 |
| **Actual 1** | 8 | 142 |

275 of 300 images classified correctly. Errors are asymmetric — 17 false positives against 8 false negatives — consistent with the decision threshold being left at 0.5 rather than tuned on the validation set.

## Dataset

[`cmudrc/3d-printed-or-not`](https://huggingface.co/datasets/cmudrc/3d-printed-or-not) from the Hugging Face Hub: 51,488 images across two classes.

Preparation:

1. Filter to images of exactly 256 × 256 pixels.
2. Randomly sample 1,500 images per class (3,000 total) using a fixed seed. Sampling is random rather than first-N because the dataset is sorted by label.
3. Stratified split into 80% train / 10% validation / 10% test (2,400 / 300 / 300).


## Method

**Classification head** (identical across all three backbones):

```
GlobalAveragePooling2D -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(1, sigmoid)
```

**Two-stage transfer learning:**

| | Frozen | Learning rate | Epochs |
| --- | --- | --- | --- |
| Stage 1 | Entire base network | 1e-3 | 3 |
| Stage 2 | All but the top layers | 1e-5 | 3 |

Stage 2 reloads the best Stage 1 checkpoint before unfreezing. Batch-normalisation layers in ResNet50 and DenseNet121 stay frozen throughout, so their ImageNet running statistics are not corrupted by updates from a batch size of 16. VGG16 contains no batch-normalisation layers.

**Layers unfrozen in Stage 2:**

| Model | Layers unfrozen | Keras layer total |
| --- | --- | --- |
| VGG16 | 4 | 19 |
| ResNet50 | 20 | 175 |
| DenseNet121 | 40 | 427 |

The counts differ because Keras enumerates pooling, activation, concatenation and batch-normalisation operations as separate layers. Each value was chosen to expose a comparable upper portion of the respective network, not an equal nominal number of layers.

**Common settings:** Adam optimiser, binary cross-entropy loss, batch size 16, inverse-frequency class weights, seed 42, architecture-specific `preprocess_input`, best checkpoint retained by validation accuracy.

## Repository structure

```
├── 3d_printed_image_classification_notebook.ipynb    # end-to-end pipeline
└── README.md
```

## Getting started

Open the notebook in Google Colab and run it top to bottom — no local setup required.

The datasets package installs in the first cell. everything else (TensorFlow, scikit-learn, pandas, matplotlib, seaborn) is preinstalled in the Colab runtime. The dataset downloads automatically from the Hugging Face Hub on first run, and the sampled subset is written to data/ as PNG.

A **GPU runtime** is recommended (Runtime → Change runtime type → T4 GPU)

## Limitations

- The test set of 300 images is too small to distinguish the models statistically. McNemar's test would be the appropriate comparison, since all models are evaluated on identical images.
- Different numbers of trainable parameters were unfrozen per architecture, so the comparison is not purely architectural.

## Authors

- **Yuval Shaanan** — implementation and fine-tuning of ResNet50 and DenseNet121, inference pipeline. project presentation.
- **Amit Shraga** — implementation and fine-tuning of VGG16, cross-model comparison and evaluation. project report.


## Acknowledgements

Dataset by the [Carnegie Mellon University Design Research Collective](https://huggingface.co/cmudrc). Pretrained weights from ImageNet via `tf.keras.applications`.
