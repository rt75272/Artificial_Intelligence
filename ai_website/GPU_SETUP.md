# GPU Setup Guide for Ultra Model Training

## Current Status
✅ TensorFlow: 2.17.1 (CUDA-enabled)  
✅ GPU: NVIDIA GeForce RTX 2080 SUPER detected and used  
✅ CUDA/cuDNN: CUDA 12.0 + cuDNN 8.9 installed via Ubuntu packages  
✅ Mixed precision: Enabled on GPU  
ℹ️ XLA: Disabled by default; can be enabled safely with a one-liner (see below)

## Requirements
- NVIDIA GPU with recent drivers (580.xx verified)
- CUDA Toolkit 12.0 (system path: /usr/lib/cuda)
- cuDNN 8.9
- Python 3.12 (venv at `AI/`), TensorFlow 2.17.1, Keras backend from TF

## How we configured it
The trainer now:
- Enables GPU memory growth to avoid OOM spikes
- Enables mixed precision (`mixed_float16`) for speed on modern GPUs
- Disables XLA JIT by default to avoid libdevice issues, but automatically exposes the CUDA path to XLA so incidental usage works

## Run it
```bash
# Use the project venv interpreter
./AI/bin/python test_gpu_config.py   # optional sanity check

# Full training
./AI/bin/python train_ultra_model.py

# Fast smoke test (tiny dataset and 1 epoch/phase)
ULTRA_SMOKE_TEST=1 ./AI/bin/python train_ultra_model.py

# Increase GPU utilization (VRAM and compute)
# 1) Larger batch size (most effective)
ULTRA_BATCH_SIZE=64 ./AI/bin/python train_ultra_model.py

# 2) Let TF pre-allocate most GPU memory (cosmetic VRAM increase; may reduce fragmentation)
ULTRA_GPU_MEMORY_GROWTH=0 ./AI/bin/python train_ultra_model.py

# 3) Optionally enable XLA JIT (can improve utilization if stable on your system)
ULTRA_ENABLE_XLA=1 ./AI/bin/python train_ultra_model.py
```

## XLA notes and troubleshooting
Some TensorFlow builds probe XLA even when JIT is off. If XLA can’t find libdevice, you may see errors like:

```
INTERNAL: libdevice not found at ./libdevice.10.bc
```

What we changed:
- The trainer auto-sets `XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda` if found, which resolves libdevice lookup on Ubuntu.

Optional: turn on XLA JIT for extra performance (only if stable on your system):
```bash
ULTRA_ENABLE_XLA=1 ./AI/bin/python train_ultra_model.py
```

Override CUDA base path if needed (rare):
```bash
ULTRA_ENABLE_XLA=1 ULTRA_XLA_CUDA_DIR=/usr/local/cuda ./AI/bin/python train_ultra_model.py
```

## Performance expectations
| Hardware | Batch Size | Time (est.) | VRAM |
|----------|------------|-------------|------|
| RTX 30xx/40xx | 32 | ~45 min | 6–8 GB |
| RTX 20xx | 24–32 | ~60 min | 5–7 GB |
| CPU (8+ cores) | 16 | 3–4 h | 4–8 GB |

Actual times vary with data and augmentations.

## Current training defaults (GPU)
- Batch size: 32  
- Epochs: 15 base + 25 fine-tune  
- Mixed precision: ON  
- XLA JIT: OFF by default (opt-in via `ULTRA_ENABLE_XLA=1`)

## Accuracy tuning tips

- MixUp augmentation (regularizes and improves generalization):
	```bash
	ULTRA_MIXUP_ALPHA=0.2 ./AI/bin/python train_ultra_model.py
	```

- Label smoothing (reduces overconfidence):
	```bash
	ULTRA_LABEL_SMOOTH=0.05 ./AI/bin/python train_ultra_model.py
	```

- Focal loss (handle class imbalance/misclassified hard examples):
	```bash
	ULTRA_FOCAL_GAMMA=2.0 ULTRA_FOCAL_ALPHA=0.25 ./AI/bin/python train_ultra_model.py
	```

- Optional real-image fine-tune (best for closing the synthetic→real gap):
	- Put images in: `data_real/train/{cats,dogs}` and `data_real/validation/{cats,dogs}`
	- Run:
		```bash
		ULTRA_REAL_DATA_DIR=data_real ./AI/bin/python train_ultra_model.py
		```
	- Or prepare from a single source directory using the helper:
		```bash
		# If your real images are in /path/to/real_images/{cats,dogs}
		./AI/bin/python tools/prepare_real_data.py --input /path/to/real_images --output data_real --val-ratio 0.2

		# Create an empty template structure to fill manually
		./AI/bin/python tools/prepare_real_data.py --create-template --output data_real
		```

- Decision threshold (tune after evaluating validation metrics):
	```bash
	ULTRA_DECISION_THRESHOLD=0.6 ./AI/bin/python train_ultra_model.py
	```

- Auto-optimize decision threshold on validation set (f1 or accuracy):
	```bash
	ULTRA_OPTIMIZE_THRESHOLD=1 ULTRA_THRESH_METRIC=f1 ./AI/bin/python train_ultra_model.py
	```

- Test-time augmentation (TTA) for evaluation-only robustness (horizontal flip):
	```bash
	ULTRA_TTA=1 ./AI/bin/python train_ultra_model.py
	```

- Input resolution (helps capture more detail if GPU allows):
	```bash
	ULTRA_IMG_SIZE=256 ./AI/bin/python train_ultra_model.py
	```