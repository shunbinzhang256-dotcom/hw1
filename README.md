# HW1 EuroSAT NumPy MLP

This implementation trains a three-layer MLP classifier in the input-hidden-output sense on EuroSAT RGB images. It uses NumPy for matrix operations and PIL for image loading. It does not use PyTorch, TensorFlow, JAX, or other autograd frameworks.

## Structure

- `data.py`: dataset scan, stratified train/val/test split, image batch loading
- `layers.py`: handwritten Linear, ReLU, Tanh, Sigmoid layers with backward passes
- `model.py`: MLP, softmax cross-entropy, checkpoint save/load
- `optim.py`: SGD and step learning-rate decay
- `train.py`: training loop, validation, best-checkpoint saving
- `search.py`: grid search over learning rate, hidden size, weight decay, activation
- `test.py`: independent test evaluation, accuracy, confusion matrix, error examples
- `visualize.py`: loss/accuracy curves, first-layer weight visualization, confusion matrix

## Environment

```bash
cd /root/hw1_solution
python3 -m pip install -r requirements.txt
```

The container already has the required packages installed.

For CUDA GPU training in this container, install CuPy:

```bash
python3 -m pip install cupy-cuda12x
```

## Train

```bash
cd /root/hw1_solution
python3 train.py \
  --data-root /root/hw1/EuroSAT_RGB \
  --out-dir /root/hw1_solution/outputs/run \
  --epochs 30 \
  --batch-size 128 \
  --hidden-dim 256 \
  --activation relu \
  --lr 0.05 \
  --weight-decay 1e-4 \
  --lr-decay 0.95
```

CUDA GPU training keeps the same handwritten forward/backward code and only replaces NumPy matrix operations with CuPy:

```bash
cd /root/hw1_solution
CUDA_VISIBLE_DEVICES=1 python3 train.py \
  --data-root /root/hw1/EuroSAT_RGB \
  --out-dir /root/hw1_solution/outputs/exp_tanh_h1024_norm_feature_aug_decay98 \
  --epochs 100 \
  --batch-size 512 \
  --hidden-dim 1024 \
  --activation tanh \
  --lr 0.05 \
  --lr-decay 0.98 \
  --weight-decay 1e-3 \
  --backend cupy \
  --normalize feature \
  --augment
```

Main outputs:

- `outputs/run/best_model.npz`
- `outputs/run/final_model.npz`
- `outputs/run/splits.json`
- `outputs/run/history.json`
- `outputs/run/training_curves.png`
- `outputs/run/first_layer_weights.png`

## Hyperparameter Search

```bash
cd /root/hw1_solution
python3 search.py \
  --data-root /root/hw1/EuroSAT_RGB \
  --out-dir /root/hw1_solution/outputs/search \
  --epochs 10 \
  --learning-rates 0.1,0.05,0.01 \
  --hidden-dims 128,256,512 \
  --weight-decays 0,0.0001,0.001 \
  --activations relu,tanh
```

For a quick debug run, add `--max-per-class 50 --max-combinations 2`.

## Test

```bash
cd /root/hw1_solution
python3 test.py \
  --data-root /root/hw1/EuroSAT_RGB \
  --checkpoint /root/hw1_solution/outputs/run/best_model.npz \
  --split-file /root/hw1_solution/outputs/run/splits.json \
  --out-dir /root/hw1_solution/outputs/run/test_outputs
```

The test script prints accuracy and a confusion matrix, and saves:

- `confusion_matrix.csv`
- `confusion_matrix.png`
- `error_examples.png`
- `test_result.json`
