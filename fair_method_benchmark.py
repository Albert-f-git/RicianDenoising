"""Reproducible cross-method benchmark for MRI Rician-noise denoising.

The script guarantees that every method receives exactly the same noisy image
for a given clean slice and sigma.  PSNR and SSIM are both averaged inside the
same foreground mask, and all models use full-slice inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from skimage.metrics import structural_similarity
from tqdm import tqdm

from models.RicianNet import RicianNet
from models.dncnn import DnCNN
from models.unet import UNet


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "evaluation_results" / "fair_method_comparison"
DEFAULT_TEST_DIR = ROOT / "data" / "processed" / "test"
DEFAULT_QUALITATIVE_IMAGE = "T1_axial_047.npy"

MODEL_SPECS = (
    {
        "name": "DnCNN",
        "kind": "dncnn",
        "weight": ROOT
        / "experiments"
        / "DnCNN_Sliding_Patch64_Stride14_AdamW_20260302_230306"
        / "model_weights.pth",
    },
    {
        "name": "RicianNet",
        "kind": "riciannet",
        "weight": ROOT
        / "experiments"
        / "Stage1"
        / "RicianNet_random_Patch128_AdamW_20260307_203419"
        / "model_weights.pth",
    },
    {
        "name": "Attention U-Net",
        "kind": "attention_unet",
        "weight": ROOT
        / "experiments"
        / "Unet_LeftAttention_Random_Patch128_epoch600_GC_SGDR_20260318_151922"
        / "model_weights_final.pth",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    )
    parser.add_argument(
        "--qualitative-noise-levels",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.30],
    )
    parser.add_argument("--qualitative-image", default=DEFAULT_QUALITATIVE_IMAGE)
    parser.add_argument("--roi-size", type=int, default=52)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional quick-test limit. The default evaluates all test images.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_clean(path: Path) -> np.ndarray:
    image = np.squeeze(np.load(path)).astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D slice, got shape {image.shape} from {path}")
    low, high = float(image.min()), float(image.max())
    if high > 1.0 or low < 0.0:
        image = (image - low) / (high - low + 1e-8)
    return np.clip(image, 0.0, 1.0)


def foreground_mask(clean: np.ndarray) -> np.ndarray:
    image_8u = np.round(clean * 255.0).astype(np.uint8)
    _, thresholded = cv2.threshold(
        image_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    pad = 30
    padded = cv2.copyMakeBorder(
        thresholded, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    bridged = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(bridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return clean > 0
    filled = np.zeros_like(padded)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return filled[pad : pad + clean.shape[0], pad : pad + clean.shape[1]] > 0


def image_seed(base_seed: int, image_name: str, sigma: float) -> int:
    token = f"{base_seed}|{image_name}|{sigma:.8f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def add_rician_noise(
    clean: np.ndarray, sigma: float, base_seed: int, image_name: str
) -> np.ndarray:
    rng = np.random.default_rng(image_seed(base_seed, image_name, sigma))
    real_noise = rng.normal(0.0, sigma, clean.shape)
    imag_noise = rng.normal(0.0, sigma, clean.shape)
    noisy = np.sqrt((clean + real_noise) ** 2 + imag_noise**2)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def create_model(kind: str) -> torch.nn.Module:
    if kind == "dncnn":
        return DnCNN()
    if kind == "riciannet":
        return RicianNet()
    if kind == "attention_unet":
        return UNet(in_channels=1, out_channels=1, use_attention=True)
    raise ValueError(f"Unknown model kind: {kind}")


def load_models(device: torch.device) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    for spec in MODEL_SPECS:
        weight = Path(spec["weight"])
        if not weight.exists():
            raise FileNotFoundError(f"Missing weight for {spec['name']}: {weight}")
        model = create_model(str(spec["kind"])).to(device)
        state = torch.load(weight, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        models[str(spec["name"])] = model
    return models


def infer_full_slice(
    model: torch.nn.Module, noisy: np.ndarray, device: torch.device
) -> np.ndarray:
    tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
    height, width = noisy.shape
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)
    with torch.inference_mode():
        output = model(tensor)
    output = output[..., :height, :width].squeeze().detach().cpu().numpy()
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def masked_metrics(
    clean: np.ndarray, estimate: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    if not np.any(mask):
        return math.nan, math.nan
    mse = float(np.mean((clean[mask] - estimate[mask]) ** 2))
    psnr = 10.0 * math.log10(1.0 / mse) if mse > 0.0 else math.inf
    _, ssim_map = structural_similarity(clean, estimate, data_range=1.0, full=True)
    return psnr, float(np.mean(ssim_map[mask]))


def select_roi(clean: np.ndarray, mask: np.ndarray, size: int) -> tuple[int, int, int, int]:
    """Select a reproducible high-detail square fully inside the image."""
    height, width = clean.shape
    size = min(size, height, width)
    gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3)
    detail = cv2.GaussianBlur(np.hypot(gx, gy) * mask.astype(np.float32), (0, 0), 2)
    margin = size // 2
    valid = np.zeros_like(mask, dtype=bool)
    valid[margin : height - (size - margin), margin : width - (size - margin)] = True
    detail[~valid] = -1
    center_y, center_x = np.unravel_index(np.argmax(detail), detail.shape)
    x0 = int(np.clip(center_x - size // 2, 0, width - size))
    y0 = int(np.clip(center_y - size // 2, 0, height - size))
    return x0, y0, size, size


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    methods = ["Noisy", *(str(spec["name"]) for spec in MODEL_SPECS)]
    sigmas = sorted({float(record["sigma"]) for record in records})
    for sigma in sigmas:
        for method in methods:
            subset = [
                record
                for record in records
                if float(record["sigma"]) == sigma and record["method"] == method
            ]
            psnr_values = np.array([float(record["psnr"]) for record in subset])
            ssim_values = np.array([float(record["ssim"]) for record in subset])
            summary.append(
                {
                    "sigma": sigma,
                    "method": method,
                    "n": len(subset),
                    "psnr_mean": float(np.mean(psnr_values)),
                    "psnr_std": float(np.std(psnr_values, ddof=1)),
                    "ssim_mean": float(np.mean(ssim_values)),
                    "ssim_std": float(np.std(ssim_values, ddof=1)),
                }
            )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_horizontal_table(
    output_dir: Path, summary: list[dict[str, object]], noise_levels: list[float]
) -> None:
    by_key = {(str(row["method"]), float(row["sigma"])): row for row in summary}
    methods = ["Noisy", *(str(spec["name"]) for spec in MODEL_SPECS)]

    csv_path = output_dir / "horizontal_method_comparison.csv"
    fieldnames = ["Method"]
    for sigma in noise_levels:
        fieldnames.extend([f"sigma_{sigma:.2f}_PSNR_dB", f"sigma_{sigma:.2f}_SSIM"])
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row: dict[str, object] = {"Method": method}
            for sigma in noise_levels:
                item = by_key[(method, sigma)]
                row[f"sigma_{sigma:.2f}_PSNR_dB"] = f"{item['psnr_mean']:.3f}"
                row[f"sigma_{sigma:.2f}_SSIM"] = f"{item['ssim_mean']:.4f}"
            writer.writerow(row)

    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{不同 Rician 噪声水平下各方法的前景区域平均 PSNR/SSIM。所有方法使用完全相同的测试切片、噪声 realization、归一化和前景掩模。}",
        r"\label{tab:fair_method_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "cc" * len(noise_levels) + "}",
        r"\toprule",
        "Method & "
        + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{${{\sigma}}={sigma:.2f}$}}" for sigma in noise_levels
        )
        + r" \\",
        " & " + " & ".join(["PSNR (dB) & SSIM"] * len(noise_levels)) + r" \\",
        r"\midrule",
    ]
    for method in methods:
        cells = []
        for sigma in noise_levels:
            item = by_key[(method, sigma)]
            cells.extend([f"{item['psnr_mean']:.2f}", f"{item['ssim_mean']:.4f}"])
        latex_lines.append(method.replace("-", r"-") + " & " + " & ".join(cells) + r" \\")
    latex_lines.extend(
        [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    )
    (output_dir / "horizontal_method_comparison.tex").write_text(
        "\n".join(latex_lines), encoding="utf-8"
    )


def render_metric_curves(
    output_dir: Path, summary: list[dict[str, object]], noise_levels: list[float]
) -> None:
    styles = {
        "Noisy": ("#7f7f7f", "x"),
        "DnCNN": ("#2a9d8f", "o"),
        "RicianNet": ("#457b9d", "s"),
        "Attention U-Net": ("#e63946", "^"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for method, (color, marker) in styles.items():
        rows = [row for row in summary if row["method"] == method]
        rows.sort(key=lambda row: float(row["sigma"]))
        axes[0].plot(
            noise_levels,
            [float(row["psnr_mean"]) for row in rows],
            label=method,
            color=color,
            marker=marker,
            linewidth=2,
        )
        axes[1].plot(
            noise_levels,
            [float(row["ssim_mean"]) for row in rows],
            label=method,
            color=color,
            marker=marker,
            linewidth=2,
        )
    axes[0].set(xlabel=r"Rician noise level $\sigma$", ylabel="Foreground PSNR (dB)")
    axes[1].set(xlabel=r"Rician noise level $\sigma$", ylabel="Foreground SSIM")
    axes[0].set_title("(a) PSNR")
    axes[1].set_title("(b) SSIM")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "psnr_ssim_vs_noise.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_qualitative(
    output_dir: Path,
    image_name: str,
    sigma: float,
    images: dict[str, np.ndarray],
    metrics: dict[str, tuple[float, float]],
    roi: tuple[int, int, int, int],
) -> None:
    order = ["GT", "Noisy", "DnCNN", "RicianNet", "Attention U-Net"]
    x0, y0, width, height = roi
    fig, axes = plt.subplots(2, len(order), figsize=(15, 6.1))
    for column, method in enumerate(order):
        image = images[method]
        axes[0, column].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0, column].add_patch(
            Rectangle((x0, y0), width, height, fill=False, edgecolor="#ff3b30", linewidth=1.5)
        )
        if method == "GT":
            title = "GT"
        else:
            psnr, ssim = metrics[method]
            title = f"{method}\n{psnr:.2f} dB / {ssim:.4f}"
        axes[0, column].set_title(title, fontsize=10.5)
        axes[0, column].axis("off")

        zoom = image[y0 : y0 + height, x0 : x0 + width]
        axes[1, column].imshow(zoom, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for spine in axes[1, column].spines.values():
            spine.set_edgecolor("#ff3b30")
            spine.set_linewidth(1.5)
        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])
        if column == 0:
            axes[1, column].set_ylabel("Zoomed ROI", fontsize=11)
    fig.suptitle(f"{image_name} — Rician noise $\\sigma={sigma:.2f}$", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(
        output_dir / f"denoising_comparison_sigma_{sigma:.2f}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    seed_everything(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.test_dir.resolve().glob("*.npy"))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No .npy images found in {args.test_dir}")
    qualitative_path = args.test_dir.resolve() / args.qualitative_image
    if not qualitative_path.exists():
        raise FileNotFoundError(f"Qualitative image not found: {qualitative_path}")
    if qualitative_path not in image_paths and args.max_images is not None:
        image_paths.append(qualitative_path)

    device = torch.device(args.device)
    models = load_models(device)
    records: list[dict[str, object]] = []
    qualitative: dict[float, dict[str, np.ndarray]] = {}
    qualitative_metrics: dict[float, dict[str, tuple[float, float]]] = {}
    start = time.perf_counter()

    for sigma in args.noise_levels:
        for image_path in tqdm(image_paths, desc=f"sigma={sigma:.2f}"):
            clean = load_clean(image_path)
            mask = foreground_mask(clean)
            noisy = add_rician_noise(clean, sigma, args.seed, image_path.name)
            noisy_metrics = masked_metrics(clean, noisy, mask)
            records.append(
                {
                    "image": image_path.name,
                    "sigma": sigma,
                    "method": "Noisy",
                    "psnr": noisy_metrics[0],
                    "ssim": noisy_metrics[1],
                }
            )
            outputs: dict[str, np.ndarray] = {}
            for name, model in models.items():
                denoised = infer_full_slice(model, noisy, device)
                outputs[name] = denoised
                psnr, ssim = masked_metrics(clean, denoised, mask)
                records.append(
                    {
                        "image": image_path.name,
                        "sigma": sigma,
                        "method": name,
                        "psnr": psnr,
                        "ssim": ssim,
                    }
                )

            if image_path.name == args.qualitative_image and any(
                math.isclose(sigma, level, abs_tol=1e-9)
                for level in args.qualitative_noise_levels
            ):
                qualitative[sigma] = {"GT": clean, "Noisy": noisy, **outputs}
                qualitative_metrics[sigma] = {
                    "Noisy": noisy_metrics,
                    **{
                        name: masked_metrics(clean, output, mask)
                        for name, output in outputs.items()
                    },
                }

    summary = summarize(records)
    write_csv(output_dir / "per_image_metrics.csv", records)
    write_csv(output_dir / "summary_metrics.csv", summary)
    write_horizontal_table(output_dir, summary, args.noise_levels)
    render_metric_curves(output_dir, summary, args.noise_levels)

    clean_for_roi = load_clean(qualitative_path)
    roi = select_roi(clean_for_roi, foreground_mask(clean_for_roi), args.roi_size)
    for sigma, images in qualitative.items():
        render_qualitative(
            output_dir,
            args.qualitative_image,
            sigma,
            images,
            qualitative_metrics[sigma],
            roi,
        )

    metadata = {
        "seed": args.seed,
        "test_directory": str(args.test_dir.resolve()),
        "test_images": len(image_paths),
        "noise_levels": args.noise_levels,
        "qualitative_image": args.qualitative_image,
        "qualitative_noise_levels": args.qualitative_noise_levels,
        "roi_xywh": roi,
        "metric_region": "foreground mask for both PSNR and SSIM",
        "inference_mode": "full slice, zero-pad right/bottom to multiple of 16",
        "device": str(device),
        "models": [
            {"name": spec["name"], "weight": str(Path(spec["weight"]).resolve())}
            for spec in MODEL_SPECS
        ],
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Benchmark complete. Results: {output_dir}")


if __name__ == "__main__":
    main()
