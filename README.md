# Learning to Reject Adversarial Examples Through Selective Prediction

CIFAR-10 experiment for COMPSCI 764. A deterministic classifier always emits the label. Four post-hoc scores decide whether to accept that label. Thresholds are chosen on clean validation data and then frozen.

## Layout

- `experiment/run_experiment.py`: training, attacks, selective metrics, and figures
- `experiment/download_cifar.py`: fetches the Hugging Face CIFAR-10 parquet files and writes `data/cifar10.pt`
- `experiment/results/summary.json`: accuracies, calibration, AURC, and paired rejection differences for the full run
- `experiment/results/operating_points.csv`: coverage, selective risk, and error-rejection rate
- `experiment/results/figures/`: plots used in the report
- `experiment/results/checkpoints/`: natural and adversarially trained weights

`experiment/results/config.json` records the full-run settings (100 natural epochs, 20 adversarial-training epochs, PGD-20, adaptive subset of 2000). The dataset is not in this repository.

## Reproduce

```bash
cd experiment
pip install -r requirements.txt
python download_cifar.py
python run_experiment.py
```

Training uses one NVIDIA GPU, seed 0, and takes roughly an hour on an RTX 5070 Laptop GPU. `python run_experiment.py --resume` reloads `results/checkpoints/natural.pt` and skips natural training.

The adaptive attack maximizes cross-entropy minus λ times the z-scored accept score. That decreases the accept score, so it is not an evasion attack against the rejector.

## Scope

CIFAR-10, one SmallCNN, one training seed. The threat model is L∞ with ε = 8/255, using FGSM and PGD. AutoAttack is not included.
