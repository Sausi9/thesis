import argparse
import csv
import json
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_experiment(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "name" not in config:
        raise ValueError(f"Experiment config {path} must define 'name'.")
    if "variants" not in config:
        raise ValueError(f"Experiment config {path} must define 'variants'.")
    return config


def git_value(args: list[str], cwd: Path, default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return default


def format_hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_override(key: str, value: Any) -> str:
    return f"{key}={format_hydra_value(value)}"


def merged_overrides(config: dict[str, Any], variant: dict[str, Any], run_name: str) -> dict[str, Any]:
    overrides = dict(config.get("base_overrides", {}))
    overrides.update(variant.get("overrides", {}))
    overrides["run_name"] = run_name
    return overrides


def build_bsub_command(
    *,
    config: dict[str, Any],
    experiment_name: str,
    variant_id: str,
    run_name: str,
    overrides: dict[str, Any],
    repo_root: Path,
) -> tuple[list[str], Path, Path]:
    job = config.get("job", {})
    log_dir = repo_root / "jobs" / "logs" / experiment_name
    stdout = log_dir / f"{variant_id}_%J.out"
    stderr = log_dir / f"{variant_id}_%J.err"

    command = [
        "bsub",
        "-q",
        str(job.get("queue", "gpua100")),
        "-J",
        run_name,
        "-n",
        str(job.get("cpus", 4)),
        "-gpu",
        str(job.get("gpu", "num=1:mode=exclusive_process")),
        "-R",
        "span[hosts=1]",
        "-R",
        f"rusage[mem={job.get('memory', '16GB')}]",
        "-M",
        str(job.get("memory", "16GB")),
        "-W",
        str(job.get("walltime", "24:00")),
        "-o",
        str(stdout.relative_to(repo_root)),
        "-e",
        str(stderr.relative_to(repo_root)),
    ]

    email = job.get("email")
    if email:
        command.extend(["-u", str(email), "-B", "-N"])

    command.extend(
        [
            "bash",
            "jobs/tds.sh",
            *[format_override(key, value) for key, value in overrides.items()],
        ]
    )
    return command, stdout, stderr


def parse_job_id(stdout: str) -> str | None:
    match = re.search(r"Job <(\d+)>", stdout)
    return match.group(1) if match else None


def manifest_paths(repo_root: Path, experiment_name: str) -> tuple[Path, Path]:
    manifest_dir = repo_root / "runs" / "experiments" / experiment_name
    return manifest_dir / "manifest.jsonl", manifest_dir / "manifest.csv"


def write_manifest(repo_root: Path, experiment_name: str, entries: list[dict[str, Any]]) -> None:
    if not entries:
        return

    jsonl_path, csv_path = manifest_paths(repo_root, experiment_name)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    fieldnames = [
        "submitted_at",
        "experiment",
        "variant",
        "run_name",
        "status",
        "job_id",
        "git_commit",
        "git_dirty",
        "sample_glob",
        "stdout",
        "stderr",
        "overrides",
        "command",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for entry in entries:
            row = {key: entry.get(key) for key in fieldnames}
            row["overrides"] = json.dumps(entry["overrides"], sort_keys=True)
            writer.writerow(row)


def submit_experiments(args: argparse.Namespace) -> None:
    repo_root = project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    config = load_experiment(config_path)
    experiment_name = str(config["name"])
    variants = config["variants"]
    only = set(args.only or [])
    submitted_at = datetime.now().isoformat(timespec="seconds")
    git_commit = git_value(["rev-parse", "HEAD"], cwd=repo_root)
    git_dirty = bool(git_value(["status", "--porcelain"], cwd=repo_root, default=""))

    entries = []
    for variant in variants:
        variant_id = str(variant["id"])
        if only and variant_id not in only:
            continue

        run_name = f"{experiment_name}_{variant_id}"
        overrides = merged_overrides(config, variant, run_name)
        command, stdout, stderr = build_bsub_command(
            config=config,
            experiment_name=experiment_name,
            variant_id=variant_id,
            run_name=run_name,
            overrides=overrides,
            repo_root=repo_root,
        )

        print(shlex.join(command))
        status = "dry_run"
        job_id = None
        if not args.dry_run:
            stdout.parent.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                command,
                cwd=repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            print(completed.stdout.strip())
            if completed.stderr.strip():
                print(completed.stderr.strip())
            job_id = parse_job_id(completed.stdout)
            status = "submitted"

        entries.append(
            {
                "submitted_at": submitted_at,
                "experiment": experiment_name,
                "variant": variant_id,
                "run_name": run_name,
                "status": status,
                "job_id": job_id,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "sample_glob": f"runs/samples/{run_name}_tds_samples_*.pt",
                "stdout": str(stdout.relative_to(repo_root)),
                "stderr": str(stderr.relative_to(repo_root)),
                "overrides": overrides,
                "command": shlex.join(command),
            }
        )

    if args.dry_run and not args.write_dry_run_manifest:
        print("Dry run only; manifest not written.")
        return

    write_manifest(repo_root, experiment_name, entries)
    jsonl_path, csv_path = manifest_paths(repo_root, experiment_name)
    print(f"Wrote manifest: {jsonl_path}")
    print(f"Wrote manifest: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a manifest-defined TDS experiment sweep.")
    parser.add_argument(
        "--config",
        default="experiments/tds_resampling_fixed.yaml",
        help="Experiment YAML path, relative to the bivariate project root unless absolute.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print bsub commands without submitting.")
    parser.add_argument(
        "--write-dry-run-manifest",
        action="store_true",
        help="Write dry-run commands to the manifest.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Optional variant ids to submit, e.g. systematic_adaptive multinomial_always.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    submit_experiments(parse_args())
