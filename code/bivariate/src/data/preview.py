from datetime import datetime
from pathlib import Path

import hydra
import matplotlib
from src.distributions.gaussian import calculate_conditional_params
import torch
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import build_dataloaders


def resolve_output_path(cfg: DictConfig) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / str(cfg.preview.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.preview.output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{cfg.dataset.name}_{timestamp}.png"
    else:
        output_name = str(cfg.preview.output_name)
        if not output_name.endswith(".png"):
            output_name = f"{output_name}.png"
    return output_dir / output_name


def take_samples(loader, num_samples: int) -> torch.Tensor:
    batches = []
    remaining = int(num_samples)
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        batches.append(x[:remaining])
        remaining -= x.shape[0]
        if remaining <= 0:
            break
    return torch.cat(batches, dim=0)


def plot_gaussian_contours(
    ax,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    *,
    color: str = "#111827",
    linestyle: str = "-",
    label: str | None = None,
) -> None:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    angles = torch.linspace(0, 2 * torch.pi, 300)
    circle = torch.stack([angles.cos(), angles.sin()])

    for radius, alpha in ((1.0, 0.9), (2.0, 0.55), (3.0, 0.3)):
        ellipse = (
            mean[:, None]
            + radius * eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ circle
        )
        ax.plot(
            ellipse[0].numpy(),
            ellipse[1].numpy(),
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            alpha=alpha,
            label=label if radius == 1.0 else None,
        )


def make_preview_figure(
    samples: torch.Tensor,
    cfg: DictConfig,
    *,
    sample_contours: bool = False,
    title_suffix: str = "preview",
    extra_contours: list[dict] | None = None,
):
    sns.set_theme(style=str(cfg.preview.style), context="notebook")
    samples = samples.detach().cpu()
    mean = samples.mean(dim=0)
    covariance = torch.cov(samples.T)

    fig = plt.figure(figsize=(9, 9), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1.1),
        height_ratios=(1.1, 4),
    )
    ax_hist_x = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[1, 0])
    ax_hist_y = fig.add_subplot(grid[1, 1], sharey=ax_scatter)
    ax_info = fig.add_subplot(grid[0, 1])

    x = samples[:, 0].numpy()
    y = samples[:, 1].numpy()

    sns.scatterplot(
        x=x,
        y=y,
        ax=ax_scatter,
        s=8,
        alpha=0.28,
        linewidth=0,
        color="#2563eb",
    )
    sns.histplot(x=x, bins=int(cfg.preview.bins), ax=ax_hist_x, color="#2563eb")
    sns.histplot(y=y, bins=int(cfg.preview.bins), ax=ax_hist_y, color="#2563eb")

    target_mean = torch.tensor(cfg.dataset.mean, dtype=torch.float32)
    target_covariance = torch.tensor(cfg.dataset.covariance, dtype=torch.float32)
    plot_gaussian_contours(
        ax_scatter,
        target_mean,
        target_covariance,
        color="#111827",
        linestyle="-",
        label="target" if sample_contours else None,
    )
    if sample_contours:
        if extra_contours is not None:
            for contour in extra_contours:
                plot_gaussian_contours(
                    ax_scatter,
                    torch.as_tensor(contour["mean"], dtype=torch.float32),
                    torch.as_tensor(contour["covariance"], dtype=torch.float32),
                    color=str(contour.get("color", "#7c3aed")),
                    linestyle=str(contour.get("linestyle", "-.")),
                    label=str(contour.get("label", "extra target")),
                )
        plot_gaussian_contours(
            ax_scatter,
            mean,
            covariance,
            color="#f97316",
            linestyle="--",
            label="samples",
        )
        ax_scatter.legend(frameon=False, loc="best")

    ax_scatter.set_title(f"{cfg.dataset.name}_{title_suffix}")
    ax_scatter.set_xlabel("x")
    ax_scatter.set_ylabel("y")
    ax_hist_x.set_xlabel("")
    ax_hist_x.set_ylabel("count")
    ax_hist_y.set_xlabel("count")
    ax_hist_y.set_ylabel("")
    ax_info.axis("off")
    ax_info.text(
        0.0,
        1.0,
        (
            f"n = {samples.shape[0]}\n"
            f"mean = [{mean[0]:.3f}, {mean[1]:.3f}]\n"
            f"cov = [[{covariance[0, 0]:.3f}, {covariance[0, 1]:.3f}],\n"
            f"       [{covariance[1, 0]:.3f}, {covariance[1, 1]:.3f}]]"
        ),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    sns.despine(fig=fig)
    return fig, mean, covariance


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    bundle = build_dataloaders(cfg.dataset, cfg.dataloader)
    samples = take_samples(bundle.train, int(cfg.preview.num_samples))
    fig, mean, covariance = make_preview_figure(samples, cfg)
    output_path = resolve_output_path(cfg)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"dataset={bundle.name} dim={bundle.dim} samples={samples.shape[0]}")
    print(f"saved preview to: {output_path}")
    print(f"sample mean: {mean.tolist()}")
    print(f"sample covariance: {covariance.tolist()}")

    calculate_conditional(cfg)


if __name__ == "__main__":
    main()
