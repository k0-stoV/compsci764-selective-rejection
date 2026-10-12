"""
Selective rejection of adversarial examples on CIFAR-10.

The emitted label is always the deterministic (dropout-off) prediction.
Selectors only decide whether to accept that label. Thresholds are chosen
on clean validation data and then frozen for clean and adversarial test data.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

SCORE_KEYS = ("msp", "temp", "mc", "comb")
SCORE_NAMES = {
    "msp": "MSP",
    "temp": "Temperature-scaled MSP",
    "mc": "MC-dropout neg. entropy",
    "comb": "Uncertainty + closeness",
}
COLORS = {
    "msp": "#0072B2",
    "temp": "#E69F00",
    "mc": "#009E73",
    "comb": "#D55E00",
}
COVERAGES = (0.95, 0.90, 0.80, 0.70, 0.60, 0.50)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        self.file.write(message + "\n")
        self.file.flush()


class SmallCNN(nn.Module):
    def __init__(self, dropout: float = 0.3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        self.hidden = nn.Sequential(nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True))
        self.dropout_h = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(self.normalize(x))
        h = torch.flatten(h, 1)
        feat = self.hidden(h)
        logits = self.classifier(self.dropout_h(feat))
        return feat, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[1]


class AugmentedSubset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray, augment: bool):
        self.base = base
        self.indices = indices
        self.augment = augment
        self.crop = transforms.RandomCrop(32, padding=4)
        self.flip = transforms.RandomHorizontalFlip()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        x, y = self.base[int(self.indices[item])]
        if self.augment:
            x = self.flip(self.crop(x))
        return x, y


def enable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(1) == y).float().mean().item())


def project(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    return x_adv.clamp(0.0, 1.0)


def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    model.eval()
    x_adv = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    return project(x_adv.detach() + eps * x_adv.grad.detach().sign(), x, eps)


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    model.eval()
    x0 = x.detach()
    delta = torch.empty_like(x0).uniform_(-eps, eps)
    x_adv = project(x0 + delta, x0, eps)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        x_adv = project(x_adv.detach() + step_size * x_adv.grad.detach().sign(), x0, eps)
    return x_adv.detach()


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    logits = logits.detach()
    labels = labels.detach()
    log_t = nn.Parameter(torch.zeros(1, device=logits.device))
    optimizer = torch.optim.Adam([log_t], lr=0.05)
    nll = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_t = 1.0
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        temperature = log_t.exp()
        loss = nll(logits / temperature, labels)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().item())
        if value < best_loss:
            best_loss = value
            best_t = float(temperature.detach().item())
    return best_t


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if mask.sum() == 0:
            continue
        total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return total


def class_prototypes(features: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, num_classes: int = 10):
    prototypes = []
    counts = []
    for cls in range(num_classes):
        mask = (labels == cls) & (preds == labels)
        counts.append(int(mask.sum().item()))
        if counts[-1] == 0:
            prototypes.append(torch.zeros(features.size(1)))
        else:
            prototypes.append(features[mask].mean(0))
    return torch.stack(prototypes), counts


def closeness(features: torch.Tensor, preds: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    mu = prototypes[preds]
    return F.cosine_similarity(features, mu, dim=1)


def threshold_for_coverage(scores: np.ndarray, coverage: float) -> float:
    n = len(scores)
    k = int(np.floor(coverage * n))
    k = min(max(k, 1), n)
    return float(np.partition(scores, n - k)[n - k])


def evaluate_at_threshold(scores: np.ndarray, correct: np.ndarray, tau: float) -> dict:
    accept = scores >= tau
    incorrect = ~correct
    n_accept = int(accept.sum())
    n_incorrect = int(incorrect.sum())
    selective_risk = float("nan") if n_accept == 0 else float(1.0 - correct[accept].mean())
    rejection_on_errors = float("nan") if n_incorrect == 0 else float((~accept[incorrect]).mean())
    return {
        "threshold": tau,
        "coverage": float(accept.mean()),
        "selective_risk": selective_risk,
        "correct_accept_rate": float((accept & correct).mean()),
        "rejection_rate_on_errors": rejection_on_errors,
        "n": int(len(scores)),
        "n_accepted": n_accept,
        "n_errors": n_incorrect,
    }


def risk_coverage_curve(scores: np.ndarray, correct: np.ndarray, n_points: int = 200):
    order = np.argsort(-scores, kind="mergesort")
    correct_sorted = correct[order].astype(np.float64)
    n = len(scores)
    cum_correct = np.cumsum(correct_sorted)
    ks = np.arange(1, n + 1)
    coverages = ks / n
    risks = 1.0 - cum_correct / ks
    aurc = float(np.trapezoid(risks, coverages))
    idx = np.linspace(0, n - 1, n_points).astype(int)
    return coverages[idx], risks[idx], aurc


def bootstrap_rate_diff(reject_a: np.ndarray, reject_b: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(reject_a)
    if n == 0:
        return {"diff": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = reject_a[idx].mean(axis=1) - reject_b[idx].mean(axis=1)
    return {
        "diff": float(reject_a.mean() - reject_b.mean()),
        "ci95_low": float(np.quantile(diffs, 0.025)),
        "ci95_high": float(np.quantile(diffs, 0.975)),
        "n": int(n),
    }


def rejection_indicators(scores: np.ndarray, correct: np.ndarray, tau: float) -> np.ndarray:
    incorrect = ~correct
    accept = scores >= tau
    return ~accept[incorrect]


class TorchRejector(nn.Module):
    def __init__(self, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray, intercept: float):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32).clamp_min(1e-8))
        self.register_buffer("coef", torch.tensor(coef, dtype=torch.float32))
        self.register_buffer("intercept", torch.tensor(float(intercept), dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        z = (features - self.mean) / self.scale
        return torch.sigmoid(z @ self.coef + self.intercept)


def collect_deterministic(model: nn.Module, loader: DataLoader, device: torch.device, log: Logger, name: str):
    model.eval()
    feats, logits, labels = [], [], []
    started = time.time()
    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        feat, logit = model.forward_features(x)
        feats.append(feat.detach().cpu())
        logits.append(logit.detach().cpu())
        labels.append(y.detach().cpu())
        if batch_idx % 40 == 0:
            log(f"  {name}: batch {batch_idx}/{len(loader)}")
    log(f"  {name} done in {time.time() - started:.1f}s, n={sum(t.size(0) for t in labels)}")
    return torch.cat(feats), torch.cat(logits), torch.cat(labels)


def collect_mc(model: nn.Module, loader: DataLoader, preds: torch.Tensor, samples: int, device: torch.device, log: Logger, name: str):
    neg_entropy, variance = [], []
    offset = 0
    started = time.time()
    for batch_idx, (x, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        batch = x.size(0)
        pred = preds[offset : offset + batch].to(device)
        offset += batch
        sum_p = torch.zeros(batch, 10, device=device)
        sum_sel = torch.zeros(batch, device=device)
        sumsq_sel = torch.zeros(batch, device=device)
        enable_dropout(model)
        with torch.no_grad():
            for _ in range(samples):
                _, logits = model.forward_features(x)
                prob = torch.softmax(logits, dim=1)
                sum_p = sum_p + prob
                selected = prob.gather(1, pred.unsqueeze(1)).squeeze(1)
                sum_sel = sum_sel + selected
                sumsq_sel = sumsq_sel + selected * selected
        model.eval()
        mean_p = sum_p / samples
        entropy = -(mean_p * mean_p.clamp_min(1e-12).log()).sum(1)
        var = (sumsq_sel / samples - (sum_sel / samples) ** 2).clamp_min(0.0)
        neg_entropy.append((-entropy).cpu())
        variance.append(var.cpu())
        if batch_idx % 20 == 0:
            log(f"  {name} MC: batch {batch_idx}/{len(loader)}")
    log(f"  {name} MC done in {time.time() - started:.1f}s")
    return torch.cat(neg_entropy), torch.cat(variance)


def generate_adversaries(model, loader, attack, device, log, name):
    model.eval()
    images, labels = [], []
    started = time.time()
    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        images.append(attack(model, x, y).cpu())
        labels.append(y.cpu())
        if batch_idx % 20 == 0:
            log(f"  {name}: batch {batch_idx}/{len(loader)}")
    log(f"  {name} done in {time.time() - started:.1f}s")
    return torch.cat(images), torch.cat(labels)


def selection_score_and_logits(model, rejector, prototypes, temperature, x, kind, mc_samples):
    """Differentiable accept-score used by the adaptive attack, plus deterministic logits."""
    model.eval()
    feat_det, logits = model.forward_features(x)
    if kind == "msp":
        return torch.softmax(logits, dim=1).max(dim=1).values, logits
    if kind == "temp":
        return torch.softmax(logits / temperature, dim=1).max(dim=1).values, logits

    pred = logits.argmax(dim=1).detach()
    batch = x.size(0)
    sum_p = torch.zeros(batch, 10, device=x.device)
    sum_sel = torch.zeros(batch, device=x.device)
    sumsq_sel = torch.zeros(batch, device=x.device)
    enable_dropout(model)
    for _ in range(mc_samples):
        _, sampled_logits = model.forward_features(x)
        prob = torch.softmax(sampled_logits, dim=1)
        sum_p = sum_p + prob
        selected = prob.gather(1, pred.unsqueeze(1)).squeeze(1)
        sum_sel = sum_sel + selected
        sumsq_sel = sumsq_sel + selected * selected
    model.eval()
    mean_p = sum_p / mc_samples
    neg_entropy = (mean_p * mean_p.clamp_min(1e-12).log()).sum(dim=1)
    var = (sumsq_sel / mc_samples - (sum_sel / mc_samples) ** 2).clamp_min(0.0)
    if kind == "mc":
        return neg_entropy, logits
    mu = prototypes[pred]
    kappa = F.cosine_similarity(feat_det, mu, dim=1)
    features = torch.stack([neg_entropy, var, kappa], dim=1)
    return rejector(features), logits


def adaptive_pgd(model, rejector, prototypes, temperature, x, y, kind, eps, steps, step_size, mc_samples, lam, score_mean, score_std):
    model.eval()
    x0 = x.detach()
    delta = torch.empty_like(x0).uniform_(-eps, eps)
    x_adv = project(x0 + delta, x0, eps)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        score, logits = selection_score_and_logits(
            model, rejector, prototypes, temperature, x_adv, kind, mc_samples
        )
        score_norm = (score - score_mean) / score_std
        loss = F.cross_entropy(logits, y) - lam * score_norm.mean()
        loss.backward()
        x_adv = project(x_adv.detach() + step_size * x_adv.grad.detach().sign(), x0, eps)
    return x_adv.detach()


def train_model(model, loader, val_loader, device, epochs, lr_drops, log, tag):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_drops, gamma=0.1)
    best_state = None
    best_acc = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        started = time.time()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * y.size(0)
            total_correct += int((logits.argmax(1) == y).sum().item())
            total += y.size(0)
        scheduler.step()
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                val_correct += int((logits.argmax(1) == y).sum().item())
                val_total += y.size(0)
        val_acc = val_correct / max(val_total, 1)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "train_acc": total_correct / max(total, 1),
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
        }
        history.append(row)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(
            f"[{tag}] epoch {epoch:03d}/{epochs} loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.4f} val_acc={val_acc:.4f} lr={row['lr']:.4g} "
            f"time={row['seconds']:.1f}s"
        )
    model.load_state_dict(best_state)
    return history, best_acc


def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )


def tensor_loader(images, labels, batch_size):
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False, num_workers=0)


def bundle_from_parts(msp, temp, neg_entropy, mc_var, kappa, clf, scaler):
    raw = np.stack([neg_entropy, mc_var, kappa], axis=1)
    combined = clf.predict_proba(scaler.transform(raw))[:, 1]
    return {"msp": msp, "temp": temp, "mc": neg_entropy, "comb": combined}


def softmax_max(logits: torch.Tensor, temperature: float) -> np.ndarray:
    return torch.softmax(logits / temperature, dim=1).max(dim=1).values.numpy()


def build_bundle(feats, logits, labels, prototypes, temperature, neg_entropy, mc_var, clf, scaler):
    preds = logits.argmax(1)
    correct = preds.eq(labels).numpy()
    kappa = closeness(feats, preds, prototypes).numpy()
    msp = softmax_max(logits, 1.0)
    temp = softmax_max(logits, temperature)
    scores = bundle_from_parts(msp, temp, neg_entropy.numpy(), mc_var.numpy(), kappa, clf, scaler)
    return {
        "scores": scores,
        "correct": correct,
        "labels": labels.numpy(),
        "preds": preds.numpy(),
        "kappa": kappa,
        "neg_entropy": neg_entropy.numpy(),
        "mc_var": mc_var.numpy(),
    }


def score_split(model, images, labels, prototypes, temperature, clf, scaler, samples, batch_size, device, log, name):
    loader = tensor_loader(images, labels, batch_size)
    feats, logits, labels_out = collect_deterministic(model, loader, device, log, name)
    preds = logits.argmax(1)
    neg_entropy, mc_var = collect_mc(model, loader, preds, samples, device, log, name)
    return build_bundle(feats, logits, labels_out, prototypes, temperature, neg_entropy, mc_var, clf, scaler)


def fit_rejector(neg_entropy, mc_var, kappa, correct, log):
    raw = np.stack([neg_entropy.numpy(), mc_var.numpy(), kappa.numpy()], axis=1)
    scaler = StandardScaler()
    x = scaler.fit_transform(raw)
    y = correct.astype(np.int64)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(x, y)
    coef = {
        "neg_entropy": float(clf.coef_[0, 0]),
        "mc_var": float(clf.coef_[0, 1]),
        "closeness": float(clf.coef_[0, 2]),
        "intercept": float(clf.intercept_[0]),
    }
    log(
        "  logistic coefficients (standardized): "
        f"neg_entropy={coef['neg_entropy']:.3f}, mc_var={coef['mc_var']:.3f}, "
        f"closeness={coef['closeness']:.3f}, intercept={coef['intercept']:.3f}"
    )
    return clf, scaler, coef


def operating_rows(tag, split_name, bundle, thresholds):
    rows = []
    for key in SCORE_KEYS:
        for coverage, tau in thresholds[key].items():
            metrics = evaluate_at_threshold(bundle["scores"][key], bundle["correct"], tau)
            metrics.update(
                {
                    "model": tag,
                    "split": split_name,
                    "score": key,
                    "target_coverage": coverage,
                }
            )
            rows.append(metrics)
    return rows


def curve_payload(bundle):
    payload = {}
    for key in SCORE_KEYS:
        coverages, risks, aurc = risk_coverage_curve(bundle["scores"][key], bundle["correct"])
        payload[key] = {
            "coverage": coverages.tolist(),
            "selective_risk": risks.tolist(),
            "aurc": aurc,
        }
    return payload


def group_means(bundle):
    correct = bundle["correct"].astype(bool)
    return {
        "n": int(len(correct)),
        "n_correct": int(correct.sum()),
        "n_incorrect": int((~correct).sum()),
        "accuracy": float(correct.mean()),
        "msp": _group_pair(bundle["scores"]["msp"], correct),
        "temp": _group_pair(bundle["scores"]["temp"], correct),
        "mc": _group_pair(bundle["scores"]["mc"], correct),
        "comb": _group_pair(bundle["scores"]["comb"], correct),
        "kappa": _group_pair(bundle["kappa"], correct),
    }


def _group_pair(values, correct):
    return {
        "correct_mean": float(values[correct].mean()) if correct.any() else float("nan"),
        "incorrect_mean": float(values[~correct].mean()) if (~correct).any() else float("nan"),
    }


def paired_comparisons(bundle, thresholds, coverages, seed):
    out = {}
    for coverage in coverages:
        entry = {}
        incorrect = ~bundle["correct"]
        for left, right in (("comb", "msp"), ("comb", "temp"), ("comb", "mc"), ("mc", "msp")):
            reject_left = rejection_indicators(bundle["scores"][left], bundle["correct"], thresholds[left][coverage])
            reject_right = rejection_indicators(bundle["scores"][right], bundle["correct"], thresholds[right][coverage])
            entry[f"{left}_minus_{right}"] = bootstrap_rate_diff(reject_left, reject_right, n_boot=1000, seed=seed)
        entry["n_errors"] = int(incorrect.sum())
        out[f"{coverage:.2f}"] = entry
    return out


def save_bundle(path: Path, bundle: dict) -> None:
    np.savez_compressed(
        path,
        correct=bundle["correct"],
        labels=bundle["labels"],
        preds=bundle["preds"],
        kappa=bundle["kappa"],
        neg_entropy=bundle["neg_entropy"],
        mc_var=bundle["mc_var"],
        **{f"score_{key}": bundle["scores"][key] for key in SCORE_KEYS},
    )


def plot_risk_coverage(curves_by_split, path: Path, title: str, ylim=(0.0, 1.0)) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key in SCORE_KEYS:
        curve = curves_by_split[key]
        ax.plot(curve["coverage"], curve["selective_risk"], color=COLORS[key], label=SCORE_NAMES[key], linewidth=1.8)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_rejection(rows, path: Path, model_tag: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=False)
    for ax, split, title, ylim in zip(
        axes,
        ("fgsm", "pgd"),
        ("FGSM", "PGD"),
        ((0.0, 1.0), (-0.002, 0.002)),
    ):
        for key in SCORE_KEYS:
            pts = [r for r in rows if r["model"] == model_tag and r["split"] == split and r["score"] == key]
            pts = sorted(pts, key=lambda r: float(r["target_coverage"]))
            ax.plot(
                [float(r["target_coverage"]) for r in pts],
                [float(r["rejection_rate_on_errors"]) for r in pts],
                marker="o",
                color=COLORS[key],
                label=SCORE_NAMES[key],
                linewidth=1.6,
            )
        ax.set_title(f"Successful {title} examples")
        ax.set_xlabel("Target clean coverage")
        ax.set_ylim(*ylim)
        ax.grid(True, linewidth=0.4, alpha=0.5)
    axes[0].set_ylabel("Rejection rate")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Rejection of successful attacks at thresholds fixed on clean validation")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_signal_means(groups, path: Path) -> None:
    signals = [("msp", "MSP"), ("kappa", "Closeness"), ("comb", "Combined score")]
    splits = [("clean", "Clean"), ("fgsm", "FGSM"), ("pgd", "PGD")]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.6), sharey=False)
    x = np.arange(len(splits))
    width = 0.36
    for ax, (signal, title) in zip(axes, signals):
        correct_vals = [groups[split][signal]["correct_mean"] for split, _ in splits]
        incorrect_vals = [groups[split][signal]["incorrect_mean"] for split, _ in splits]
        ax.bar(x - width / 2, correct_vals, width, label="Classifier correct", color="#0072B2")
        ax.bar(x + width / 2, incorrect_vals, width, label="Classifier wrong", color="#D55E00")
        ax.set_xticks(x, [name for _, name in splits])
        ax.set_title(title)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    axes[0].set_ylabel("Mean score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Score means by whether the deterministic classifier is correct", y=1.14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_operating_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "model",
        "split",
        "score",
        "target_coverage",
        "threshold",
        "coverage",
        "selective_risk",
        "correct_accept_rate",
        "rejection_rate_on_errors",
        "n",
        "n_accepted",
        "n_errors",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


class MemoryCIFAR(Dataset):
    """Official CIFAR-10 stored as uint8 NCHW tensors. Items are float images in [0, 1]."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        image = self.images[index].float().div(255.0)
        return image, int(self.labels[index])


def prepare_splits(root: Path, seed: int, limit: int | None):
    path = root / "cifar10.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run download_cifar.py first.")
    blob = torch.load(path, weights_only=False)
    base_train = MemoryCIFAR(blob["train_x"], blob["train_y"])
    base_test = MemoryCIFAR(blob["test_x"], blob["test_y"])
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(base_train), generator=generator).numpy()
    train_idx, calib_idx, val_idx = perm[:40000], perm[40000:45000], perm[45000:]
    test_idx = np.arange(len(base_test))
    if limit is not None:
        train_idx = train_idx[:limit]
        calib_idx = calib_idx[: max(limit // 5, 32)]
        val_idx = val_idx[: max(limit // 5, 32)]
        test_idx = test_idx[: max(limit // 2, 32)]
    return base_train, base_test, train_idx, calib_idx, val_idx, test_idx


def load_split_tensors(base, indices):
    chosen = torch.as_tensor(np.asarray(indices), dtype=torch.long)
    return base.images[chosen].float().div(255.0), base.labels[chosen].long()


def run_attack_suite(model, clean_images, clean_labels, prototypes, temperature, clf, scaler, args, device, log, tag, outdir: Path):
    eval_bs = args.batch_size * 2
    samples = args.mc_samples
    result_dir = outdir / tag
    result_dir.mkdir(parents=True, exist_ok=True)

    log(f"[{tag}] scoring clean data")
    clean = score_split(
        model, clean_images, clean_labels, prototypes, temperature, clf, scaler,
        samples, eval_bs, device, log, f"{tag}-clean",
    )
    save_bundle(result_dir / "clean.npz", clean)

    set_seed(args.seed + 11)
    log(f"[{tag}] generating FGSM")
    fgsm_images, fgsm_labels = generate_adversaries(
        model,
        tensor_loader(clean_images, clean_labels, args.batch_size),
        lambda m, x, y: fgsm_attack(m, x, y, args.eps),
        device,
        log,
        f"{tag}-fgsm",
    )
    log(f"[{tag}] scoring FGSM")
    fgsm = score_split(
        model, fgsm_images, fgsm_labels, prototypes, temperature, clf, scaler,
        samples, eval_bs, device, log, f"{tag}-fgsm",
    )
    save_bundle(result_dir / "fgsm.npz", fgsm)

    set_seed(args.seed + 17)
    log(f"[{tag}] generating PGD")
    pgd_images, pgd_labels = generate_adversaries(
        model,
        tensor_loader(clean_images, clean_labels, args.batch_size),
        lambda m, x, y: pgd_attack(m, x, y, args.eps, args.pgd_steps, args.pgd_step_size),
        device,
        log,
        f"{tag}-pgd",
    )
    log(f"[{tag}] scoring PGD")
    pgd = score_split(
        model, pgd_images, pgd_labels, prototypes, temperature, clf, scaler,
        samples, eval_bs, device, log, f"{tag}-pgd",
    )
    save_bundle(result_dir / "pgd.npz", pgd)
    return {"clean": clean, "fgsm": fgsm, "pgd": pgd}, pgd_images


def run_adaptive(model, clf, scaler, prototypes, temperature, val_bundle, clean_images, clean_labels, thresholds, args, device, log, tag):
    rejector = TorchRejector(scaler.mean_, scaler.scale_, clf.coef_.reshape(-1), float(clf.intercept_[0])).to(device)
    prototypes = prototypes.to(device)
    n = len(clean_labels)
    subset_size = min(args.adaptive_subset, n)
    rng = np.random.default_rng(args.seed + 23)
    subset = np.sort(rng.choice(n, size=subset_size, replace=False))
    x_sub = clean_images[subset]
    y_sub = clean_labels[subset]
    score_stats = {}
    for kind in SCORE_KEYS:
        values = val_bundle["scores"][kind]
        score_stats[kind] = {"mean": float(values.mean()), "std": float(max(values.std(), 1e-6))}

    log(f"[{tag}] adaptive PGD on {subset_size} images, steps={args.adaptive_steps}, lambda={args.adaptive_lambda}")
    per_score = {}
    rows = []
    for index, kind in enumerate(SCORE_KEYS):
        set_seed(args.seed + 30 + index)
        started = time.time()
        images = []
        for x, y in tensor_loader(x_sub, y_sub, args.adaptive_batch_size):
            x = x.to(device)
            y = y.to(device)
            images.append(
                adaptive_pgd(
                    model,
                    rejector,
                    prototypes,
                    temperature,
                    x,
                    y,
                    kind,
                    args.eps,
                    args.adaptive_steps,
                    args.pgd_step_size,
                    args.adaptive_mc_samples,
                    args.adaptive_lambda,
                    score_stats[kind]["mean"],
                    score_stats[kind]["std"],
                ).cpu()
            )
        adv_images = torch.cat(images)
        log(f"  generated adaptive-{kind} in {time.time() - started:.1f}s; scoring")
        bundle = score_split(
            model, adv_images, y_sub, prototypes.cpu(), temperature, clf, scaler,
            args.mc_samples, args.batch_size * 2, device, log, f"{tag}-adaptive-{kind}",
        )
        per_score[kind] = bundle
        metrics = evaluate_at_threshold(bundle["scores"][kind], bundle["correct"], thresholds[kind][0.80])
        metrics.update({"attack": kind, "target_coverage": 0.80, "attack_success_rate": float((~bundle["correct"]).mean())})
        rows.append(metrics)
        metrics60 = evaluate_at_threshold(bundle["scores"][kind], bundle["correct"], thresholds[kind][0.60])
        metrics60.update({"attack": kind, "target_coverage": 0.60, "attack_success_rate": float((~bundle["correct"]).mean())})
        rows.append(metrics60)
        log(
            f"  adaptive-{kind}: success={metrics['attack_success_rate']:.3f} "
            f"reject@0.80={metrics['rejection_rate_on_errors']:.3f} "
            f"reject@0.60={metrics60['rejection_rate_on_errors']:.3f}"
        )
    return {"subset_size": int(subset_size), "subset_indices": subset.tolist(), "operating_points": rows}


def evaluate_model(model, base_train, train_idx, calib_images, calib_labels, val_images, val_labels, test_images, test_labels, args, device, log, tag, outdir: Path):
    eval_bs = args.batch_size * 2
    log(f"[{tag}] deterministic features for prototypes and calibration")
    train_loader = tensor_loader(*load_split_tensors(base_train, train_idx), eval_bs)
    calib_loader = tensor_loader(calib_images, calib_labels, eval_bs)
    val_loader = tensor_loader(val_images, val_labels, eval_bs)
    train_feats, train_logits, train_labels = collect_deterministic(model, train_loader, device, log, f"{tag}-train")
    calib_feats, calib_logits, calib_y = collect_deterministic(model, calib_loader, device, log, f"{tag}-calib")
    val_feats, val_logits, val_y = collect_deterministic(model, val_loader, device, log, f"{tag}-val")

    temperature = fit_temperature(calib_logits.to(device), calib_y.to(device))
    log(f"[{tag}] temperature = {temperature:.4f}")
    train_preds = train_logits.argmax(1)
    prototypes, proto_counts = class_prototypes(train_feats, train_labels, train_preds)
    log(f"[{tag}] prototype counts = {proto_counts}")

    log(f"[{tag}] MC-dropout on calibration and validation")
    calib_neg, calib_var = collect_mc(model, calib_loader, calib_logits.argmax(1), args.mc_samples, device, log, f"{tag}-calib")
    val_neg, val_var = collect_mc(model, val_loader, val_logits.argmax(1), args.mc_samples, device, log, f"{tag}-val")
    calib_kappa = closeness(calib_feats, calib_logits.argmax(1), prototypes)
    val_kappa = closeness(val_feats, val_logits.argmax(1), prototypes)
    calib_correct = calib_logits.argmax(1).eq(calib_y).numpy()
    clf, scaler, coefficients = fit_rejector(calib_neg, calib_var, calib_kappa, calib_correct, log)

    val_bundle = build_bundle(val_feats, val_logits, val_y, prototypes, temperature, val_neg, val_var, clf, scaler)
    clean_conf = softmax_max(calib_logits, 1.0)
    temp_conf = softmax_max(calib_logits, temperature)
    ece = {
        "msp_calib": expected_calibration_error(clean_conf, calib_correct),
        "temp_calib": expected_calibration_error(temp_conf, calib_correct),
    }
    auroc = {key: float(roc_auc_score(val_bundle["correct"], val_bundle["scores"][key])) for key in SCORE_KEYS}
    log(f"[{tag}] clean-val AUROC of correctness: {json.dumps({k: round(v, 4) for k, v in auroc.items()})}")

    thresholds = {}
    for key in SCORE_KEYS:
        thresholds[key] = {coverage: threshold_for_coverage(val_bundle["scores"][key], coverage) for coverage in COVERAGES}

    bundles, pgd_images = run_attack_suite(
        model, test_images, test_labels, prototypes, temperature, clf, scaler, args, device, log, tag, outdir
    )
    test_logits_conf = bundles["clean"]["scores"]["msp"]
    test_temp_conf = bundles["clean"]["scores"]["temp"]
    ece["msp_test"] = expected_calibration_error(test_logits_conf, bundles["clean"]["correct"])
    ece["temp_test"] = expected_calibration_error(test_temp_conf, bundles["clean"]["correct"])

    rows = []
    curves = {}
    groups = {}
    comparisons = {}
    for split_name, bundle in bundles.items():
        rows.extend(operating_rows(tag, split_name, bundle, thresholds))
        curves[split_name] = curve_payload(bundle)
        groups[split_name] = group_means(bundle)
        if split_name != "clean":
            comparisons[split_name] = paired_comparisons(bundle, thresholds, (0.80, 0.60), args.seed)
        log(
            f"[{tag}] {split_name} accuracy={bundle['correct'].mean():.4f} "
            f"AURC comb={curves[split_name]['comb']['aurc']:.4f} msp={curves[split_name]['msp']['aurc']:.4f}"
        )

    adaptive = None
    if args.adaptive_subset > 0:
        adaptive = run_adaptive(
            model, clf, scaler, prototypes, temperature, val_bundle,
            test_images, test_labels, thresholds, args, device, log, tag,
        )
        # Same-subset non-adaptive PGD reference for the two main coverages.
        # Reconstructed by regenerating PGD on the stored subset with the same seed as full PGD
        # is not identical image-wise to a slice, so slice the already generated full PGD set.
        subset = np.array(adaptive["subset_indices"])
        pgd_subset = {
            "scores": {key: bundles["pgd"]["scores"][key][subset] for key in SCORE_KEYS},
            "correct": bundles["pgd"]["correct"][subset],
        }
        reference_rows = []
        for kind in SCORE_KEYS:
            for coverage in (0.80, 0.60):
                metrics = evaluate_at_threshold(pgd_subset["scores"][kind], pgd_subset["correct"], thresholds[kind][coverage])
                metrics.update({
                    "attack": f"pgd_subset_scored_by_{kind}",
                    "target_coverage": coverage,
                    "attack_success_rate": float((~pgd_subset["correct"]).mean()),
                })
                reference_rows.append(metrics)
        adaptive["matched_pgd_subset"] = reference_rows

    summary = {
        "tag": tag,
        "temperature": temperature,
        "prototype_counts": proto_counts,
        "logistic_coefficients_standardized": coefficients,
        "ece": ece,
        "val_auroc_correctness": auroc,
        "accuracy": {name: float(bundle["correct"].mean()) for name, bundle in bundles.items()},
        "aurc": {name: {key: curves[name][key]["aurc"] for key in SCORE_KEYS} for name in curves},
        "group_means": groups,
        "paired_rejection_diff_on_errors": comparisons,
        "adaptive": adaptive,
    }
    return summary, rows, curves, groups, bundles


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--adv-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--pgd-step-size", type=float, default=2 / 255)
    parser.add_argument("--adaptive-steps", type=int, default=10)
    parser.add_argument("--adaptive-subset", type=int, default=2000)
    parser.add_argument("--adaptive-batch-size", type=int, default=64)
    parser.add_argument("--adaptive-mc-samples", type=int, default=4)
    parser.add_argument("--adaptive-lambda", type=float, default=1.0)
    parser.add_argument("--skip-advtrain", action="store_true")
    parser.add_argument("--skip-adaptive", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.adv_epochs = 1
        args.mc_samples = 2
        args.pgd_steps = 2
        args.adaptive_steps = 1
        args.adaptive_subset = 32
        args.adaptive_mc_samples = 2
        args.adaptive_batch_size = 16
        limit = 256
    else:
        limit = None
    if args.skip_adaptive:
        args.adaptive_subset = 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    log = Logger(args.outdir / "log.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    log(f"device={device}")
    if device.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}")

    base_train, base_test, train_idx, calib_idx, val_idx, test_idx = prepare_splits(args.data_dir, args.seed, limit)
    log(f"splits train={len(train_idx)} calib={len(calib_idx)} val={len(val_idx)} test={len(test_idx)}")
    train_loader = make_loader(AugmentedSubset(base_train, train_idx, augment=True), args.batch_size, True, args.seed)
    # Checkpoint selection uses the calibration split so the validation split is used only for thresholds.
    calib_loader = make_loader(AugmentedSubset(base_train, calib_idx, augment=False), args.batch_size, False, args.seed)
    calib_images, calib_labels = load_split_tensors(base_train, calib_idx)
    val_images, val_labels = load_split_tensors(base_train, val_idx)
    test_images, test_labels = load_split_tensors(base_test, test_idx)

    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config.update({
        "eps": args.eps,
        "architecture": "SmallCNN dropout 0.3, 3x conv-pairs, 256-d penultimate features",
        "optimizer": "SGD lr=0.1 momentum=0.9 weight_decay=5e-4",
        "lr_schedule_natural": "x0.1 at epochs 50 and 75",
        "prediction": "deterministic argmax; selectors do not change the label",
        "checkpoint": "best calibration-split accuracy during training; validation split is reserved for thresholds",
        "threshold_rule": "on clean validation, tau is the k-th largest score for k=floor(coverage*n)",
        "combined_score": "logistic regression on calib: MC neg-entropy, variance of p(y_hat), cosine to predicted-class prototype",
        "threat_model": "Linf eps=8/255, FGSM and PGD-20 step 2/255; adaptive PGD maximizes CE - lambda * z-scored accept score",
    })
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    log("training natural model")
    model = SmallCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"parameters={n_params}")
    ckpt_dir = args.outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    natural_ckpt = ckpt_dir / "natural.pt"
    if args.resume and natural_ckpt.exists():
        saved = torch.load(natural_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        best_val = float(saved["best_val"])
        log(f"resumed natural checkpoint, best val accuracy={best_val:.4f}")
    else:
        history, best_val = train_model(model, train_loader, calib_loader, device, args.epochs, [50, 75], log, "natural")
        torch.save({"state_dict": model.state_dict(), "history": history, "best_val": best_val}, natural_ckpt)
        log(f"best natural val accuracy={best_val:.4f}")

    natural_summary, rows, curves, groups, _ = evaluate_model(
        model, base_train, train_idx, calib_images, calib_labels, val_images, val_labels,
        test_images, test_labels, args, device, log, "natural", args.outdir,
    )
    fig_dir = args.outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_risk_coverage(curves["clean"], fig_dir / "risk_coverage_clean.png", "Clean CIFAR-10 test, natural model", ylim=(0.0, 0.16))
    plot_risk_coverage(curves["pgd"], fig_dir / "risk_coverage_pgd.png", "PGD test, threshold swept on PGD scores", ylim=(0.98, 1.02))
    plot_rejection(rows, fig_dir / "rejection_vs_coverage.png", "natural")
    plot_signal_means(groups, fig_dir / "score_means_by_correctness.png")
    write_operating_csv(args.outdir / "operating_points.csv", rows)
    (args.outdir / "curves_natural.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    (args.outdir / "summary.json").write_text(
        json.dumps({"natural": _jsonify(natural_summary), "advtrain": None, "config": config}, indent=2),
        encoding="utf-8",
    )
    log("saved natural-model results")

    adv_summary = None
    if not args.skip_advtrain:
        log("training adversarially (PGD-7) model")
        adv_model = SmallCNN().to(device)
        adv_loader = make_loader(AugmentedSubset(base_train, train_idx, augment=True), args.batch_size, True, args.seed + 3)

        def adv_train():
            optimizer = torch.optim.SGD(adv_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            milestones = [m for m in (10, 15) if m < args.adv_epochs]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones or [args.adv_epochs + 1], gamma=0.1)
            best_state = None
            best_acc = -1.0
            history_adv = []
            for epoch in range(1, args.adv_epochs + 1):
                adv_model.train()
                total_loss = 0.0
                total_correct = 0
                total = 0
                started = time.time()
                for x, y in adv_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    adv_model.eval()
                    x_adv = pgd_attack(adv_model, x, y, args.eps, steps=7 if not args.smoke else 1, step_size=args.pgd_step_size)
                    adv_model.train()
                    optimizer.zero_grad(set_to_none=True)
                    logits = adv_model(x_adv)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * y.size(0)
                    total_correct += int((logits.argmax(1) == y).sum().item())
                    total += y.size(0)
                scheduler.step()
                val_correct = 0
                val_total = 0
                adv_model.eval()
                with torch.no_grad():
                    for x, y in calib_loader:
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        val_correct += int((adv_model(x).argmax(1) == y).sum().item())
                        val_total += y.size(0)
                val_acc = val_correct / max(val_total, 1)
                history_adv.append({"epoch": epoch, "train_loss": total_loss / max(total, 1), "train_acc": total_correct / max(total, 1), "val_acc": val_acc, "seconds": time.time() - started})
                if val_acc >= best_acc:
                    best_acc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in adv_model.state_dict().items()}
                log(f"[advtrain] epoch {epoch:03d}/{args.adv_epochs} loss={total_loss / max(total, 1):.4f} val_acc={val_acc:.4f} time={time.time() - started:.1f}s")
            adv_model.load_state_dict(best_state)
            return history_adv, best_acc

        history_adv, best_adv = adv_train()
        torch.save({"state_dict": adv_model.state_dict(), "history": history_adv, "best_val": best_adv}, ckpt_dir / "advtrain.pt")
        saved_subset = args.adaptive_subset
        args.adaptive_subset = 0
        adv_summary, adv_rows, adv_curves, _, _ = evaluate_model(
            adv_model, base_train, train_idx, calib_images, calib_labels, val_images, val_labels,
            test_images, test_labels, args, device, log, "advtrain", args.outdir,
        )
        args.adaptive_subset = saved_subset
        rows.extend(adv_rows)
        plot_risk_coverage(adv_curves["clean"], fig_dir / "risk_coverage_clean_advtrain.png", "Clean test, adversarially trained model")
        plot_risk_coverage(adv_curves["pgd"], fig_dir / "risk_coverage_pgd_advtrain.png", "PGD test, adversarially trained model")

    write_operating_csv(args.outdir / "operating_points.csv", rows)
    payload = {"natural": _jsonify(natural_summary), "advtrain": _jsonify(adv_summary), "config": config}
    (args.outdir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.outdir / "curves_natural.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    log("finished")
    log(f"results written to {args.outdir.resolve()}")


def _jsonify(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


if __name__ == "__main__":
    main()
