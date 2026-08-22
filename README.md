# Forgery Detection in JPEG Images

**Authors**: Divit Sharma, Shivam Singh, Anurag Jaiswal, Srilatha

This repository contains our implementation of block-level double JPEG compression detection for image forgery localization. The project explores various deep learning architectures and configurations to identify tampered regions in digital images by analyzing their compression artifacts.

---

## Project Overview

When images are edited using tools like Photoshop, tampered regions undergo single compression (decompress → edit → compress), while untouched areas retain their original double compression. Our approach detects this discrepancy by analyzing raw DCT coefficients extracted directly from the JPEG bitstream, avoiding the quantization errors introduced by working with decompressed pixel data.

---

## Repository Structure

```
├── CODES/
│   ├── Densenet121.ipynb           # DenseNet-121 implementation (256x256 patches)
│   ├── resnet50.ipynb              # ResNet-50 architecture variant
│   ├── cnnnew.ipynb                # Custom CNN model from the paper
│   ├── patch_128x128.ipynb         # DenseNet with 128x128 patch size
│   ├── Without_q_factor.ipynb      # Ablation study without Q-factor features
│   ├── Patch_Localization.ipynb    # Full forgery localization pipeline
│   ├── UCID_Columbia.ipynb         # Testing on UCID and Columbia datasets
│   ├── quantization_matrix.py      # Script to generate test dataset with unseen Q-tables
│   └── q_table_list.mat            # 504 unseen quantization matrices for testing
│
├── Output/
│   ├── output1.jpeg                # Result visualization (original | heatmap | mask)
│   ├── output2.jpeg                # Result visualization (original | heatmap | mask)
│   └── output3.jpeg                # Result visualization (original | heatmap | mask)
│
├── PPT/
│   └── DIP_Presentation.pptx       # Project presentation slides
│
├── DIP_Report.pdf                  # Detailed project report
└── README.md                       # This file
```

---

## Code Files Explained

### Main Training Notebooks

**`Densenet121.ipynb`**  
Primary implementation using DenseNet-121 architecture with 256x256 patches. Extracts 64×161×2 feature tensors (DCT histograms + Q-factors) and trains on GPU with the full dataset. Includes train/val/test splits matching the paper's methodology.

**`resnet50.ipynb`**  
Alternative implementation using ResNet-50 as the backbone network. Same feature extraction pipeline but different network architecture for comparison.

**`cnnnew.ipynb`**  
Implements the exact CNN architecture described in the original paper. Uses four convolutional blocks with max pooling, followed by fully connected layers that incorporate quantization table information.

### Experimental Variants

**`patch_128x128.ipynb`**  
Modified version using 128×128 patches instead of 256×256. Tests whether smaller patch sizes can maintain detection accuracy while reducing computational cost.

**`Without_q_factor.ipynb`**  
Ablation study that excludes quantization factors from the feature vector. Helps understand the contribution of Q-table information to model performance.

### Testing and Localization

**`Patch_Localization.ipynb`**  
Complete pipeline for forgery localization on full images. Performs sliding window inference, generates heatmaps, applies Gaussian smoothing, and produces binary forgery masks.

**`UCID_Columbia.ipynb`**  
Evaluation script for testing trained models on the UCID (1338 uncompressed color images) and Columbia datasets using the 504 unseen quantization matrices.

### Dataset Preparation

**`quantization_matrix.py`**  
Python script to generate the test dataset. Loads 504 unseen quantization matrices from the `.mat` file, extracts 256×256 non-overlapping patches from raw images, and creates both single-compressed and double-compressed versions for evaluation.

---

## Datasets Used

- **HuggingFace Dataset**: 1.14M patches for training/validation (streamed during runtime to avoid memory issues)
- **UCID Dataset**: 1338 uncompressed images for generating test patches with unseen quantization matrices
- **Columbia Dataset**: Additional uncompressed images for generalization testing

---

## Feature Extraction Pipeline

1. **DCT Coefficient Extraction**: Uses `jpegio` library to read quantized DCT coefficients directly from JPEG files
2. **Histogram Generation**: For each of the 64 frequency positions (8×8 DCT blocks), creates a histogram of coefficient values in range [-80, 80]
3. **Q-Factor Concatenation**: Appends the corresponding quantization step size from the Q-table
4. **Final Feature Tensor**: 64 × 161 × 2 (64 frequencies × 161 histogram bins × 2 channels)

---

## Model Architectures

### DenseNet-121
- Custom DenseNet with dense blocks and transition layers
- Input: 64×161×2 feature maps
- Growth rate: k=32
- Four dense blocks with [6, 12, 24, 16] layers
- Output: Binary classification (single vs. double compressed)

### ResNet-50
- Standard ResNet architecture adapted for JPEG feature inputs
- Residual connections help with gradient flow
- Similar input preprocessing as DenseNet

### Paper CNN
- Four convolutional blocks (32→64→128→128 channels)
- Q-table information injected at fully connected layers
- Designed specifically for histogram-based inputs

---

## Training Configuration

- **Optimizer**: Adam with initial LR = 0.0001
- **Scheduler**: Learning rate increases to 0.0005 at epoch 15
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: Varies by available GPU memory
- **Device**: CUDA GPU (automatically selected if available)
- **Metrics**: Accuracy, TPR (True Positive Rate), TNR (True Negative Rate)

---

## Output Visualizations

The `Output/` folder contains comparison images showing:
- **Left**: Original test image
- **Middle**: Forgery probability heatmap (warmer colors indicate higher probability of tampering)
- **Right**: Binary forgery mask after thresholding

---

## Installation and Usage

### Prerequisites
```bash
pip install torch torchvision jpegio numpy scipy opencv-python matplotlib tqdm
```

### Running Training
1. Open any of the training notebooks (`Densenet121.ipynb`, `resnet50.ipynb`, etc.)
2. Update dataset paths if necessary
3. Run all cells sequentially
4. Model checkpoints will be saved as `best_model.pth`

### Running Localization
1. Open `Patch_Localization.ipynb`
2. Load a trained model checkpoint
3. Provide test image path
4. Run inference to generate heatmap and forgery mask

### Generating Test Dataset
1. Download UCID/Columbia images
2. Update paths in `quantization_matrix.py`
3. Run the script to generate single/double compressed patches
```bash
python quantization_matrix.py
```

---

## Key Dependencies

- **jpegio**: Low-level JPEG manipulation (DCT coefficient extraction)
- **PyTorch**: Deep learning framework
- **NumPy/SciPy**: Numerical operations and filtering
- **OpenCV**: Image processing utilities
- **tqdm**: Progress bars for long-running operations

---

## Results

Detailed experimental results, accuracy metrics, confusion matrices, and comparative analysis across different architectures are documented in `DIP_Report.pdf`.

---

## References

This implementation is based on the paper:  
**"Block-level double JPEG compression detection for image forgery localization"**  
Vinay Verma, Deepak Singh, Nitin Khanna (2024)

---

## Notes

- All models expect grayscale JPEG images (Y-channel only)
- Patch extraction is non-overlapping during training, sliding window during inference
- Gaussian smoothing with σ=5 is applied to heatmaps before thresholding
- The 504 test quantization matrices are deliberately unseen during training
