"""Compare the existing deep models with the Chen and Liu numerical methods.

The deep-model records are reused from ``fair_method_comparison`` after their
metadata are validated.  Chen and Liu receive the exact same deterministic
noisy slices and are evaluated with the same foreground PSNR/SSIM functions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import special
from tqdm import tqdm

import fair_method_benchmark as common


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = ROOT / "evaluation_results" / "fair_method_comparison"
DEFAULT_OUTPUT_DIR = ROOT / "evaluation_results" / "fair_deep_numerical_comparison"

DEEP_METHODS = ("DnCNN", "RicianNet", "Attention U-Net")
CHEN_NAME = "Chen-Zeng (2015)"
LIU_NAME = "Liu et al. (2022)"
METHODS = ("Noisy", *DEEP_METHODS, CHEN_NAME, LIU_NAME)

METHOD_INFO = {
    "Noisy": "Input baseline",
    "DnCNN": "Deep learning",
    "RicianNet": "Deep learning (Li et al., 2020)",
    "Attention U-Net": "Deep learning",
    CHEN_NAME: "Numerical / variational",
    LIU_NAME: "Numerical / variational",
}

LIU_PAPER_PARAMETER_SETS = {
    15.0: {"alpha": 0.015, "beta": 0.080},
    25.0: {"alpha": 0.010, "beta": 0.045},
    35.0: {"alpha": 0.005, "beta": 0.030},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=common.DEFAULT_TEST_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    )
    parser.add_argument(
        "--qualitative-noise-levels", type=float, nargs="+", default=[0.10, 0.20, 0.30]
    )
    parser.add_argument("--qualitative-image", default=common.DEFAULT_QUALITATIVE_IMAGE)
    parser.add_argument("--roi-size", type=int, default=52)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--chen-iterations", type=int, default=2000)
    parser.add_argument("--chen-min-iterations", type=int, default=25)
    parser.add_argument("--chen-tolerance", type=float, default=1e-5)
    parser.add_argument("--liu-iterations", type=int, default=500)
    parser.add_argument(
        "--recompute-chen",
        action="store_true",
        help="Recompute Chen instead of reusing compatible records in the output directory.",
    )
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def gradient(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.zeros_like(u)
    dy = np.zeros_like(u)
    dx[:, :-1] = u[:, 1:] - u[:, :-1]
    dy[:-1, :] = u[1:, :] - u[:-1, :]
    return dx, dy


def divergence(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    result = np.zeros_like(dx)
    result[:, 1:-1] = dx[:, 1:-1] - dx[:, :-2]
    result[:, 0] = dx[:, 0]
    result[:, -1] = -dx[:, -2]
    result[1:-1, :] += dy[1:-1, :] - dy[:-2, :]
    result[0, :] += dy[0, :]
    result[-1, :] -= dy[-2, :]
    return result


def masked_gradient(
    u: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward differences on the foreground domain with no-flux boundaries."""
    gx, gy = gradient(u)
    valid_x = np.zeros_like(mask, dtype=bool)
    valid_y = np.zeros_like(mask, dtype=bool)
    valid_x[:, :-1] = mask[:, :-1] & mask[:, 1:]
    valid_y[:-1, :] = mask[:-1, :] & mask[1:, :]
    gx[~valid_x] = 0.0
    gy[~valid_y] = 0.0
    return gx, gy, valid_x, valid_y


def bessel_ratio_and_derivative(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable B=I1/I0 and B' needed by Chen--Zeng equation (19)."""
    i0 = special.ive(0, x)
    denominator = np.maximum(i0, 1e-12)
    ratio = special.ive(1, x) / denominator
    ratio_prime = (i0 + special.ive(2, x)) / (2.0 * denominator) - ratio**2
    return ratio, ratio_prime


def chen_denoise(
    noisy: np.ndarray,
    sigma: float,
    max_iter: int,
    mask: np.ndarray,
    tolerance: float = 1e-5,
    min_iter: int = 25,
) -> tuple[np.ndarray, int, bool]:
    """Chen--Zeng Algorithm 1 on the segmented foreground domain.

    The paper replaces Omega by Omega/Omega_B for medical images with a zero
    background.  Differences crossing the support boundary are therefore
    removed, which implements the corresponding no-flux TV boundary.
    """
    if max_iter < 1 or min_iter < 1 or min_iter > max_iter:
        raise ValueError("Chen iteration limits must satisfy 1 <= min_iter <= max_iter")
    if tolerance <= 0.0:
        raise ValueError("Chen tolerance must be positive")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != noisy.shape or not np.any(mask):
        raise ValueError("Chen foreground mask must be nonempty and match the image")
    f = noisy.astype(np.float64) * 255.0
    f[~mask] = 0.0
    sigma_8bit = sigma * 255.0
    gamma = 0.035
    beta = 8.0 / gamma
    tau = 0.015 / gamma
    u = f.copy()
    u_bar = f.copy()
    px = np.zeros_like(f)
    py = np.zeros_like(f)
    valid_x = np.zeros_like(mask, dtype=bool)
    valid_y = np.zeros_like(mask, dtype=bool)
    valid_x[:, :-1] = mask[:, :-1] & mask[:, 1:]
    valid_y[:-1, :] = mask[:-1, :] & mask[1:, :]
    f_foreground = f[mask]
    converged = False
    iterations = max_iter
    for iteration in range(1, max_iter + 1):
        old = u.copy()
        gx, gy = gradient(u_bar)
        gx[~valid_x] = 0.0
        gy[~valid_y] = 0.0
        px += beta * gamma * gx
        py += beta * gamma * gy
        magnitude = np.maximum(1.0, np.hypot(px, py))
        px /= magnitude
        py /= magnitude
        px[~valid_x] = 0.0
        py[~valid_y] = 0.0
        div_p = divergence(px, py)
        u_safe = np.maximum(old[mask], 1e-8)
        x = f_foreground * u_safe / sigma_8bit**2
        ratio, ratio_prime = bessel_ratio_and_derivative(x)
        g_prime = (
            u_safe / sigma_8bit**2
            - f_foreground * ratio / sigma_8bit**2
            + (1.0 / sigma_8bit) * (1.0 - np.sqrt(f_foreground / u_safe))
        )
        g_second = (
            1.0 / sigma_8bit**2
            - (f_foreground / sigma_8bit**2) ** 2 * ratio_prime
            + np.sqrt(f_foreground) / (2.0 * sigma_8bit * u_safe**1.5)
        )
        residual = g_prime - gamma * div_p[mask]
        u = old.copy()
        u[mask] = np.clip(
            old[mask] - residual / (g_second + 1.0 / tau), 0.0, 255.0
        )
        u_bar = 2.0 * u - old

        if iteration >= min_iter:
            delta = u[mask] - old[mask]
            relative_change = math.sqrt(float(np.sum(delta * delta))) / max(
                math.sqrt(float(np.sum(u[mask] * u[mask]))), 1e-12
            )
            if relative_change < tolerance:
                converged = True
                iterations = iteration
                break

    estimated_clean = np.sqrt(np.maximum(f**2 - 1.2 * sigma_8bit**2, 0.0))
    u[mask] += float(np.mean(estimated_clean[mask]) - np.mean(u[mask]))
    u[~mask] = 0.0
    restored = np.clip(u / 255.0, 0.0, 1.0).astype(np.float32)
    return restored, iterations, converged


def liu_paper_parameters(sigma: float) -> tuple[float, float, float]:
    """Return the nearest parameter set explicitly reported in Liu et al."""
    sigma_8bit = sigma * 255.0
    reference_sigma = min(LIU_PAPER_PARAMETER_SETS, key=lambda value: abs(value - sigma_8bit))
    values = LIU_PAPER_PARAMETER_SETS[reference_sigma]
    return float(values["alpha"]), float(values["beta"]), reference_sigma


def liu_denoise(
    noisy: np.ndarray,
    sigma: float,
    max_iter: int,
    rng: np.random.Generator,
    alpha: float,
    beta: float,
    penalty: float = 1.0,
    c: float = 2.0,
) -> np.ndarray:
    """Liu--Chang--Duan (2022) Algorithm 4.1."""
    f = noisy.astype(np.float64) * 255.0
    sigma_8bit = sigma * 255.0
    g = np.sqrt(np.maximum(f**2 - c * sigma_8bit**2, 0.0))
    u = g.copy()
    n1 = np.zeros_like(f)
    n2 = rng.normal(0.0, sigma_8bit, f.shape)
    for _ in range(max_iter):
        old = u.copy()
        coeff = 1.0 - alpha / penalty
        v1_hat = u + coeff * n1
        v2_hat = coeff * n2
        v_norm = np.hypot(v1_hat, v2_hat)
        nonzero = v_norm > 1e-12
        safe_norm = np.maximum(v_norm, 1e-12)
        v1 = np.where(nonzero, f * v1_hat / safe_norm, f)
        v2 = np.where(nonzero, f * v2_hat / safe_norm, 0.0)

        beta_hat = beta + penalty
        u_hat = (beta * g + penalty * v1 + (alpha - penalty) * n1) / beta_hat
        px = np.zeros_like(u_hat)
        py = np.zeros_like(u_hat)
        for _ in range(2):
            div_p = divergence(px, py)
            gx, gy = gradient(div_p - beta_hat * u_hat)
            norm = np.hypot(gx, gy)
            px = (px + 0.1 * gx) / (1.0 + 0.1 * norm)
            py = (py + 0.1 * gy) / (1.0 + 0.1 * norm)
        u = u_hat - divergence(px, py) / beta_hat
        n1 = (alpha * n1 + penalty * (v1 - u)) / (alpha + penalty)
        n2 = (alpha * n2 + penalty * v2) / (alpha + penalty)
        denominator = max(float(np.linalg.norm(old)), 1e-12)
        if float(np.linalg.norm(u - old)) / denominator < 1e-4:
            break
    return np.clip(u / 255.0, 0.0, 1.0).astype(np.float32)


def process_numerical_case(
    path_text: str,
    sigma: float,
    seed: int,
    chen_iterations: int,
    chen_min_iterations: int,
    chen_tolerance: float,
    liu_iterations: int,
    qualitative_image: str,
    run_chen: bool,
) -> dict[str, object]:
    """Run one deterministic image/noise case in a worker process."""
    path = Path(path_text)
    clean = common.load_clean(path)
    mask = common.foreground_mask(clean)
    noisy = common.add_rician_noise(clean, sigma, seed, path.name)
    chen = None
    used_iterations = 0
    converged = False
    if run_chen:
        chen, used_iterations, converged = chen_denoise(
            noisy,
            sigma,
            chen_iterations,
            mask,
            tolerance=chen_tolerance,
            min_iter=chen_min_iterations,
        )
    alpha, beta, reference_sigma = liu_paper_parameters(sigma)
    liu_rng = np.random.default_rng(common.image_seed(seed + 1, path.name, sigma))
    liu = liu_denoise(
        noisy, sigma, liu_iterations, liu_rng, alpha=alpha, beta=beta, penalty=1.0, c=2.0
    )
    liu_psnr, liu_ssim = common.masked_metrics(clean, liu, mask)
    result: dict[str, object] = {
        "image": path.name,
        "sigma": sigma,
        "liu_psnr": liu_psnr,
        "liu_ssim": liu_ssim,
        "liu_reference_sigma": reference_sigma,
        "liu_alpha": alpha,
        "liu_beta": beta,
        "chen_iterations": used_iterations,
        "chen_converged": converged,
    }
    if chen is not None:
        chen_psnr, chen_ssim = common.masked_metrics(clean, chen, mask)
        result["chen_psnr"] = chen_psnr
        result["chen_ssim"] = chen_ssim
    if path.name == qualitative_image:
        result["qualitative"] = (clean, noisy, chen, liu)
    return result


def read_source_records(
    source_dir: Path,
    expected_names: set[str],
    noise_levels: list[float],
    seed: int,
) -> list[dict[str, object]]:
    metadata = json.loads((source_dir / "benchmark_metadata.json").read_text(encoding="utf-8"))
    source_noise_levels = {float(level) for level in metadata["noise_levels"]}
    if metadata["seed"] != seed or not set(noise_levels).issubset(source_noise_levels):
        raise ValueError("Source benchmark seed/noise levels do not cover this run")
    rows: list[dict[str, object]] = []
    with (source_dir / "per_image_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["image"] in expected_names
                and float(row["sigma"]) in noise_levels
                and row["method"] in ("Noisy", *DEEP_METHODS)
            ):
                rows.append(
                    {
                        "image": row["image"],
                        "sigma": float(row["sigma"]),
                        "method": row["method"],
                        "psnr": float(row["psnr"]),
                        "ssim": float(row["ssim"]),
                    }
                )
    expected = len(expected_names) * len(noise_levels) * (1 + len(DEEP_METHODS))
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} reusable records, found {len(rows)}")
    return rows


def read_existing_method_records(
    output_dir: Path,
    expected_names: set[str],
    noise_levels: list[float],
    method: str,
) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    """Load a complete compatible method block from an earlier deterministic run."""
    metrics_path = output_dir / "per_image_metrics.csv"
    metadata_path = output_dir / "benchmark_metadata.json"
    if not metrics_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    with metrics_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["image"] in expected_names
                and float(row["sigma"]) in noise_levels
                and row["method"] == method
            ):
                rows.append(
                    {
                        "image": row["image"],
                        "sigma": float(row["sigma"]),
                        "method": method,
                        "psnr": float(row["psnr"]),
                        "ssim": float(row["ssim"]),
                    }
                )
    expected = len(expected_names) * len(noise_levels)
    if len(rows) != expected:
        return None
    return rows, metadata


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sigmas = sorted({float(row["sigma"]) for row in records})
    for sigma in sigmas:
        for method in METHODS:
            subset = [
                row for row in records if float(row["sigma"]) == sigma and row["method"] == method
            ]
            psnr = np.asarray([float(row["psnr"]) for row in subset])
            ssim = np.asarray([float(row["ssim"]) for row in subset])
            rows.append(
                {
                    "sigma": sigma,
                    "method": method,
                    "method_type": METHOD_INFO[method],
                    "n": len(subset),
                    "psnr_mean": float(np.mean(psnr)),
                    "psnr_std": float(np.std(psnr, ddof=1)) if len(psnr) > 1 else 0.0,
                    "ssim_mean": float(np.mean(ssim)),
                    "ssim_std": float(np.std(ssim, ddof=1)) if len(ssim) > 1 else 0.0,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_horizontal(output_dir: Path, summary: list[dict[str, object]], sigmas: list[float]) -> None:
    lookup = {(str(row["method"]), float(row["sigma"])): row for row in summary}
    fields = ["Method", "Method type"]
    for sigma in sigmas:
        fields += [f"sigma_{sigma:.2f}_PSNR_dB", f"sigma_{sigma:.2f}_SSIM"]
    rows = []
    for method in METHODS:
        row: dict[str, object] = {"Method": method, "Method type": METHOD_INFO[method]}
        for sigma in sigmas:
            item = lookup[(method, sigma)]
            row[f"sigma_{sigma:.2f}_PSNR_dB"] = f"{item['psnr_mean']:.3f}"
            row[f"sigma_{sigma:.2f}_SSIM"] = f"{item['ssim_mean']:.4f}"
        rows.append(row)
    write_csv(output_dir / "horizontal_deep_numerical_comparison.csv", rows)

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{深度学习与数值 Rician 去噪方法在统一测试协议下的前景区域平均 PSNR/SSIM。}",
        r"\label{tab:deep_numerical_comparison}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "cc" * len(sigmas) + "}", r"\toprule",
        "Method & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{${{\sigma}}={s:.2f}$}}" for s in sigmas) + r" \\",
        " & " + " & ".join(["PSNR (dB) & SSIM"] * len(sigmas)) + r" \\", r"\midrule",
    ]
    for method in METHODS:
        values = []
        for sigma in sigmas:
            item = lookup[(method, sigma)]
            values += [f"{item['psnr_mean']:.2f}", f"{item['ssim_mean']:.4f}"]
        lines.append(method.replace("&", r"\&") + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    (output_dir / "horizontal_deep_numerical_comparison.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def render_curves(output_dir: Path, summary: list[dict[str, object]]) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    styles = {
        "Noisy": ("#7f7f7f", "x", "--"), "DnCNN": ("#2a9d8f", "o", "-"),
        "RicianNet": ("#457b9d", "s", "-"), "Attention U-Net": ("#e63946", "^", "-"),
        CHEN_NAME: ("#f4a261", "D", "-."), LIU_NAME: ("#8338ec", "P", "-."),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for method in METHODS:
        rows = sorted((row for row in summary if row["method"] == method), key=lambda x: float(x["sigma"]))
        color, marker, linestyle = styles[method]
        x = [float(row["sigma"]) for row in rows]
        axes[0].plot(x, [float(row["psnr_mean"]) for row in rows], label=method, color=color, marker=marker, linestyle=linestyle, linewidth=2)
        axes[1].plot(x, [float(row["ssim_mean"]) for row in rows], label=method, color=color, marker=marker, linestyle=linestyle, linewidth=2)
    axes[0].set(xlabel=r"莱斯噪声水平 $\sigma$", ylabel="前景 PSNR (dB)", title="(a) PSNR")
    axes[1].set(xlabel=r"莱斯噪声水平 $\sigma$", ylabel="前景 SSIM", title="(b) SSIM")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output_dir / "psnr_ssim_deep_vs_numerical.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_qualitative(output_dir: Path, image_name: str, sigma: float, images: dict[str, np.ndarray], metrics: dict[str, tuple[float, float]], roi: tuple[int, int, int, int]) -> None:
    order = ("GT", *METHODS)
    x0, y0, width, height = roi
    fig, axes = plt.subplots(2, len(order), figsize=(21, 6.1))
    for column, method in enumerate(order):
        image = images[method]
        axes[0, column].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0, column].add_patch(Rectangle((x0, y0), width, height, fill=False, edgecolor="#ff3b30", linewidth=1.4))
        axes[0, column].set_title("GT" if method == "GT" else f"{method}\n{metrics[method][0]:.2f} dB / {metrics[method][1]:.4f}", fontsize=9)
        axes[0, column].axis("off")
        axes[1, column].imshow(image[y0:y0+height, x0:x0+width], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, column].set_xticks([]); axes[1, column].set_yticks([])
        for spine in axes[1, column].spines.values():
            spine.set_edgecolor("#ff3b30"); spine.set_linewidth(1.4)
    fig.suptitle(f"{image_name} — Rician noise $\\sigma={sigma:.2f}$", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / f"deep_numerical_comparison_sigma_{sigma:.2f}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    common.seed_everything(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.test_dir.resolve().glob("*.npy"))
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]
    qualitative_path = args.test_dir.resolve() / args.qualitative_image
    if args.max_images is not None and qualitative_path not in image_paths:
        image_paths.append(qualitative_path)
    if not image_paths or not qualitative_path.exists():
        raise FileNotFoundError("Test data or qualitative image is missing")

    names = {path.name for path in image_paths}
    records = read_source_records(args.source_dir.resolve(), names, args.noise_levels, args.seed)
    existing_chen = None
    if not args.recompute_chen:
        existing_chen = read_existing_method_records(
            output_dir, names, args.noise_levels, CHEN_NAME
        )
        if existing_chen is not None:
            prior_metadata = existing_chen[1]
            compatible = (
                int(prior_metadata.get("seed", -1)) == args.seed
                and int(prior_metadata.get("test_images", -1)) == len(image_paths)
                and set(map(float, prior_metadata.get("noise_levels", [])))
                == set(map(float, args.noise_levels))
            )
            if not compatible:
                existing_chen = None
    reuse_chen = existing_chen is not None
    if reuse_chen:
        records.extend(existing_chen[0])
    qualitative_outputs: dict[float, dict[str, np.ndarray]] = {}
    chen_iteration_counts: list[int] = []
    chen_converged_count = 0
    start = time.perf_counter()
    jobs = [
        (
            str(path), sigma, args.seed, args.chen_iterations,
            args.chen_min_iterations, args.chen_tolerance, args.liu_iterations,
            args.qualitative_image,
            (not reuse_chen) or (
                path.name == args.qualitative_image
                and any(
                    math.isclose(sigma, level, abs_tol=1e-9)
                    for level in args.qualitative_noise_levels
                )
            ),
        )
        for sigma in args.noise_levels
        for path in image_paths
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_numerical_case, *job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="numerical benchmark"):
            result = future.result()
            image_name = str(result["image"])
            sigma = float(result["sigma"])
            if not reuse_chen:
                records.append(
                    {"image": image_name, "sigma": sigma, "method": CHEN_NAME,
                     "psnr": result["chen_psnr"], "ssim": result["chen_ssim"]}
                )
                chen_iteration_counts.append(int(result["chen_iterations"]))
                chen_converged_count += int(bool(result["chen_converged"]))
            records.append(
                {"image": image_name, "sigma": sigma, "method": LIU_NAME,
                 "psnr": result["liu_psnr"], "ssim": result["liu_ssim"]}
            )
            if "qualitative" in result and any(
                math.isclose(sigma, level, abs_tol=1e-9)
                for level in args.qualitative_noise_levels
            ):
                clean, noisy, chen, liu = result["qualitative"]
                qualitative_outputs[sigma] = {
                    "GT": clean, "Noisy": noisy, CHEN_NAME: chen, LIU_NAME: liu
                }

    summary = summarize(records)
    write_csv(output_dir / "per_image_metrics.csv", records)
    write_csv(output_dir / "summary_metrics.csv", summary)
    write_horizontal(output_dir, summary, args.noise_levels)
    render_curves(output_dir, summary)

    device = common.torch.device(args.device if common.torch.cuda.is_available() else "cpu")
    models = common.load_models(device)
    clean = common.load_clean(qualitative_path)
    mask = common.foreground_mask(clean)
    roi = common.select_roi(clean, mask, args.roi_size)
    for sigma, images in qualitative_outputs.items():
        noisy = images["Noisy"]
        for name, model in models.items():
            images[name] = common.infer_full_slice(model, noisy, device)
        metrics = {name: common.masked_metrics(clean, image, mask) for name, image in images.items() if name != "GT"}
        render_qualitative(output_dir, args.qualitative_image, sigma, images, metrics, roi)

    if reuse_chen:
        chen_metadata = dict(existing_chen[1]["numerical_parameters"][CHEN_NAME])
        chen_metadata["reused_from_previous_deterministic_run"] = True
    else:
        chen_metadata = {
            "gamma": 0.035,
            "maximum_iterations": args.chen_iterations,
            "minimum_iterations": args.chen_min_iterations,
            "relative_change_tolerance": args.chen_tolerance,
            "foreground_domain": "GT-derived support mask with no-flux TV boundary",
            "bias_correction_c": 1.2,
            "mean_iterations": float(np.mean(chen_iteration_counts)),
            "minimum_used_iterations": min(chen_iteration_counts),
            "maximum_used_iterations": max(chen_iteration_counts),
            "converged_cases": chen_converged_count,
            "total_cases": len(chen_iteration_counts),
        }
    liu_schedule = {}
    for sigma in args.noise_levels:
        alpha, beta, reference_sigma = liu_paper_parameters(sigma)
        liu_schedule[f"{sigma:.2f}"] = {
            "sigma_8bit": sigma * 255.0,
            "nearest_paper_sigma": reference_sigma,
            "alpha": alpha,
            "beta": beta,
        }
    metadata = {
        "seed": args.seed, "test_directory": str(args.test_dir.resolve()), "test_images": len(image_paths),
        "noise_levels": args.noise_levels, "qualitative_image": args.qualitative_image,
        "qualitative_noise_levels": args.qualitative_noise_levels, "roi_xywh": roi,
        "metric_region": "same GT-derived foreground mask for PSNR and SSIM",
        "shared_input": "same deterministic noisy realization as fair_method_comparison",
        "reused_deep_metrics_from": str(args.source_dir.resolve()),
        "reused_chen_metrics": reuse_chen,
        "method_types": METHOD_INFO,
        "numerical_parameters": {
            CHEN_NAME: chen_metadata,
            LIU_NAME: {
                "parameter_source": "Liu et al. (2022), Figure 2 caption",
                "mapping_rule": "nearest reported 8-bit sigma among 15, 25, and 35; clamp outside range",
                "schedule": liu_schedule,
                "r": 1.0,
                "maximum_iterations": args.liu_iterations,
                "relative_change_tolerance": 1e-4,
                "chambolle_inner_iterations": 2,
                "adaptive_g_c": 2.0,
            },
        },
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output_dir / "benchmark_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Benchmark complete. Results: {output_dir}")


if __name__ == "__main__":
    main()
