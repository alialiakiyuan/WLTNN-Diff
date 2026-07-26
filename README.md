# WLTNN-Diff: Convergence-Guaranteed Plug-and-Play Image Restoration via Implicit Diffusion Prior and Explicit Tensor Low-Rankness

Built upon [DiffPIR](https://github.com/yuanzhi-zhu/DiffPIR) by Yuanzhi Zhu et al.

## Environment Setup
### 1. Create and activate conda environment

```bash
conda create -n diffpir python=3.8
conda activate diffpir
```

### 2. Install dependencies

```bash
# PyTorch (CUDA 11.8)
pip install torch==2.1.0+cu118 torchvision==0.16.0+cpu torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Core scientific computing
pip install numpy==1.24.4 scipy==1.10.0 scikit-image==0.21.0 scikit-learn==1.3.2

# Image processing
pip install opencv-python==4.7.0.68 pillow==10.4.0 imageio==2.35.1

# Diffusion model support
pip install lpips==0.1.4 blobfile==2.0.1

# Tensor decomposition
pip install tensorly==0.9.0

# Utilities
pip install matplotlib==3.6.3 tqdm==4.67.1 pyyaml==6.0.2 h5py==3.11.0
pip install pandas==2.0.3 openpyxl==3.1.5
pip install omegaconf==2.1.2
```

### 3. Download model checkpoint

Download the pre-trained diffusion model from [DiffPIR model zoo](https://github.com/yuanzhi-zhu/DiffPIR/blob/main/model_zoo/README.md) and place it in `./model_zoo/`.

## Usage

1. Place clean test images in the `./testset/` folder (PNG format recommended).

2. Configure noise parameters in `demo.py`:

```python
noise_level_img = 0.03    # Gaussian noise std (in [0, 1] range)
sparse_density = 0.05     # Salt-and-pepper noise density
enable_sparse = True      # True: Gaussian + S&P noise with S-term
                          # False: Gaussian noise only, no S-term
```

3. Run the denoising experiment:

```bash
python demo.py
```

4. Results (denoised images, noisy images, metrics) are saved in `./results/`.

## Project Structure

```
├── demo.py              # Main denoising script (modified from DiffPIR)
├── trpca_l1.py          # Tensor Robust PCA with L1 proximal operator
├── guided_diffusion/    # Diffusion model (from DiffPIR)
├── utils/               # Utility functions (from DiffPIR, modified)
├── model_zoo/           # Pre-trained model checkpoints
├── configs/             # YAML config files (from DiffPIR)
├── testset/             # Clean test images (user-provided)
└── results/             # Output results
```

## Full Package List (diffpir conda environment)

<details>
<summary>Click to expand</summary>

```
antlr4-python3-runtime==4.8
anyio==4.2.0
blobfile==2.0.1
clarabel==0.9.0
contourpy==1.1.1
cvxopt==1.3.2
cvxpy==1.5.2
cycler==0.12.1
ecos==2.0.14
fancyimpute==0.7.0
fonttools==4.57.0
h5py==3.11.0
hdf5storage==0.1.19
image-quality==1.2.7
imageio==2.35.1
joblib==1.4.2
kiwisolver==1.4.7
knnimpute==0.1.0
lazy-loader==0.4
libsvm==3.23.0.4
lpips==0.1.4
lxml==4.9.4
matplotlib==3.6.3
mpi4py==4.0.3
mpmath==1.3.0
networkx==3.1
nose==1.3.7
numpy==1.24.4
omegaconf==2.1.2
opencv-contrib-python==4.7.0.72
opencv-python==4.7.0.68
opencv-python-headless==4.7.0.72
openpyxl==3.1.5
osqp==1.0.3
packaging==24.2
pandas==2.0.3
patsy==1.0.1
pillow==10.4.0
pluggy==1.5.0
pycryptodomex==3.22.0
pyparsing==3.1.4
pytest==8.3.5
python-dateutil==2.9.0
pytz==2025.2
pywavelets==1.4.1
pyyaml==6.0.2
requests==2.32.3
scikit-image==0.21.0
scikit-learn==1.3.2
scipy==1.10.0
scs==3.2.7.post2
six==1.17.0
statsmodels==0.14.1
sympy==1.13.3
tensorly==0.9.0
threadpoolctl==3.5.0
tifffile==2023.7.10
tomli==2.2.1
torch==2.1.0+cu118
torchaudio==2.1.0+cu118
torchvision==0.16.0+cpu
tqdm==4.67.1
typing-extensions==4.13.2
xlsxwriter==3.2.3
```

</details>

## Acknowledgments

This project is based on [DiffPIR](https://github.com/yuanzhi-zhu/DiffPIR):

```bibtex
@inproceedings{zhu2023denoising,
  title={Denoising Diffusion Models for Plug-and-Play Image Restoration},
  author={Yuanzhi Zhu and Kai Zhang and Jingyun Liang and Jiezhang Cao and Bihan Wen and Radu Timofte and Luc Van Gool},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2023}
}
```

The TRPCA module incorporates low-rank tensor decomposition with L1 sparse regularization for handling mixed noise.

## Citation

If you use this code, please cite both the original DiffPIR and this project:

**WLTNN-Diff** (paper in preparation):
```bibtex
% Citation will be updated upon arXiv publication.
% For now, please cite the GitHub repository:
@software{wltnn_diff,
  title     = {WLTNN-Diff: Image Denoising with Diffusion Model and TRPCA},
  url       = {https://github.com/alialiakiyuan/WLTNN-Diff},
  year      = {2026},
}
```
