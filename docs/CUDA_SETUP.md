# CUDA Setup Guide for PyTorch

This guide explains how to install PyTorch with CUDA support to enable GPU acceleration for embeddings.

## Why CUDA?

Using CUDA (GPU) instead of CPU can significantly speed up embedding generation:

- **CPU**: ~50-100 embeddings/second (varies by model)
- **GPU**: ~500-2000+ embeddings/second (varies by GPU and model)

For the `intfloat/e5-base-v2` model with 768 dimensions, GPU acceleration can be 10-20x faster.

## Prerequisites

1. **NVIDIA GPU** with CUDA support (check compatibility: https://developer.nvidia.com/cuda-gpus)
2. **NVIDIA CUDA drivers** installed (usually comes with NVIDIA drivers)
3. **CUDA Toolkit** compatible version (11.8 or 12.1 are common)

## Check Current PyTorch Installation

First, check what version you have:

```bash
python -c "import torch; cuda_ver = torch.version.cuda if torch.cuda.is_available() else 'N/A'; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {cuda_ver}')"
```

If it shows `CUDA available: False`, you need to install the CUDA-enabled version.

## Installation Methods

### Method 1: Official PyTorch Website (Recommended)

1. **Visit**: https://pytorch.org/get-started/locally/
2. **Select your configuration**:
   - PyTorch Build: Stable (recommended)
   - Your OS: Windows/Linux/macOS
   - Package: pip or conda
   - Language: Python
   - Compute Platform: CUDA 11.8 or CUDA 12.1 (check your GPU compatibility)
3. **Copy and run the command** shown on the website

**Example for CUDA 11.8 (Windows/Linux):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Example for CUDA 12.1 (Windows/Linux):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Method 2: Check Your CUDA Version First

If you're not sure which CUDA version to use:

**Windows:**

```bash
nvidia-smi
```

Look for "CUDA Version" in the top right corner of the output.

**Linux:**

```bash
nvcc --version
```

Then install the matching PyTorch CUDA version from https://pytorch.org/get-started/locally/

### Method 3: Using Conda (Alternative)

If you prefer conda:

```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Verify Installation

After installation, verify CUDA is working:

```bash
python -c "import torch; gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {gpu_name}')"
```

You should see:

- `CUDA available: True`
- CUDA version number
- GPU count and name

## Uninstall CPU-Only Version First (If Needed)

If you already have PyTorch installed (CPU-only), you may need to uninstall it first:

```bash
pip uninstall torch torchvision torchaudio
```

Then install the CUDA version using one of the methods above.

## Troubleshooting

### "CUDA not available" after installation

1. **Check NVIDIA drivers are installed**:

   ```bash
   nvidia-smi
   ```

   If this fails, install NVIDIA drivers from https://www.nvidia.com/drivers/

2. **Check PyTorch CUDA version matches your system**:

   - Your GPU must support the CUDA version
   - Your NVIDIA drivers must support the CUDA version
   - PyTorch must be built for that CUDA version

3. **Verify PyTorch can see CUDA**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.version.cuda)  # Should show version number
   ```

### "Torch not compiled with CUDA enabled"

This means you installed the CPU-only version. Uninstall and reinstall with CUDA support using the methods above.

### Performance Issues

- Make sure your `.env` file has `EMBEDDING_DEVICE=cuda`
- Check GPU memory usage with `nvidia-smi` while running embeddings
- For large batches, you may need to reduce `EMBEDDING_BATCH_SIZE` if you run out of GPU memory

## Fallback to CPU

The application automatically falls back to CPU if CUDA is requested but not available. You'll see a warning message in the logs. This ensures the application still works, just slower.

## Configuration

Once CUDA is installed and verified, set in your `backend/.env`:

```env
EMBEDDING_DEVICE=cuda
EMBEDDING_MODEL=intfloat/e5-base-v2
EMBEDDING_BATCH_SIZE=32
```

The application will automatically detect and use CUDA if available.
