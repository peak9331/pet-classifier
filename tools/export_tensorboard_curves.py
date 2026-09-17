"""只读导出 TensorBoard 曲线；日志归属或完整性不明确时停止，不猜测数据。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERIES = ("Loss/Train", "Loss/Validation", "Accuracy/Train", "Accuracy/Validation")


def inspect_run(run_dir):
    """逐事件文件读取，不创建 SummaryWriter，不修改原始日志。"""
    values = {name: [] for name in SERIES}
    source_files = []
    print(f"发现实验目录：{run_dir}")
    for event_file in sorted(run_dir.rglob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0}).Reload()
        tags = accumulator.Tags()
        print(f"  {event_file.relative_to(run_dir)}：{tags}")
        source_files.append(str(event_file))
        for tag in tags.get("scalars", []):
            # 兼容 add_scalar 的完整标签与 add_scalars 的子目录布局。
            key = tag if tag in SERIES else None
            subdir = event_file.parent.relative_to(run_dir).as_posix()
            for expected in SERIES:
                metric, split = expected.split("/")
                if tag == metric and subdir == f"{metric}_{split}":
                    key = expected
            if key is not None:
                values[key].extend((event.step, event.value) for event in accumulator.Scalars(tag))
    return {"directory": run_dir, "values": values, "files": source_files}


def check_run(run, checkpoint, expected_epochs):
    """四条曲线必须完整、不重复，并匹配 checkpoint 中的最佳轮次和验证指标。"""
    expected_steps = list(range(1, expected_epochs + 1))
    values = run["values"]
    for key, points in values.items():
        points.sort()
        if [step for step, _ in points] != expected_steps:
            return f"{key} 的轮次不完整或重复：{[step for step, _ in points]}"
        if any(not math.isfinite(value) for _, value in points):
            return f"{key} 存在非有限数值"
        if key.startswith("Accuracy") and any(not 0 <= value <= 1 for _, value in points):
            return f"{key} 不在已知的 0～1 准确率范围内"
    best_epoch = int(checkpoint["epoch"])
    # 训练在严格变好时保存，准确率并列时保留首次达到最优的轮次。
    recorded_best = max(values["Accuracy/Validation"], key=lambda point: point[1])
    if recorded_best[0] != best_epoch:
        return f"最佳 Epoch 不匹配（日志 {recorded_best[0]}，模型 {best_epoch}）"
    for key, checkpoint_key in [("Accuracy/Validation", "val_accuracy"), ("Loss/Validation", "val_loss")]:
        value = dict(values[key])[best_epoch]
        # TensorBoard 的 float32 数值允许极小舍入误差。
        if not math.isclose(value, checkpoint[checkpoint_key], rel_tol=0, abs_tol=5e-7):
            return f"{key} 与 checkpoint 不匹配"
    return None


def select_run(runs, checkpoint, expected_epochs, explicit_dir, prefixes):
    candidates = []
    for run in runs:
        path = run["directory"]
        if explicit_dir is not None:
            if path.resolve() != explicit_dir.resolve():
                continue
        elif not any(path.name.startswith(prefix) for prefix in prefixes):
            continue
        reason = check_run(run, checkpoint, expected_epochs)
        if reason is None:
            candidates.append(run)
        else:
            print(f"排除 {path.name}：{reason}")
    if len(candidates) != 1:
        raise ValueError(
            f"满足条件的实验有 {len(candidates)} 个，无法唯一确定。"
            "请核对上方目录/标签，并用 --baseline-run / --label-smoothing-run 指定。"
        )
    return candidates[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--label-smoothing-run", type=Path)
    parser.add_argument("--epochs", type=int, default=10, help="日志必须完整覆盖 1～N 轮")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "training_curves.png")
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs 必须大于 0")
    try:
        if not args.runs_dir.is_dir():
            raise ValueError(f"日志目录不存在：{args.runs_dir}")
        runs = [inspect_run(path) for path in sorted(args.runs_dir.iterdir()) if path.is_dir()]
        if not runs:
            raise ValueError("没有发现实验目录，不生成曲线。")
        selected = []
        for name, filename, explicit, prefixes in [
            ("Baseline CE", "baseline_ce_best.pth", args.baseline_run, ("baseline_",)),
            ("Label Smoothing 0.1", "label_smoothing_01_best.pth", args.label_smoothing_run, ("label_smoothing_01_",)),
        ]:
            checkpoint_path = PROJECT_ROOT / "checkpoints" / filename
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            run = select_run(runs, checkpoint, args.epochs, explicit, prefixes)
            selected.append((name, run, checkpoint_path))
            print(f"确认 {name}：{run['directory']}")
        if args.output.exists():
            print(f"保留已有曲线：{args.output}；需要新图时指定新的 --output。")
            return
        source_path = args.output.with_suffix(".sources.json")
        if source_path.exists():
            raise ValueError(f"来源记录已存在：{source_path}；请指定新的 --output。")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        provenance = []
        try:
            for column, (name, run, checkpoint_path) in enumerate(selected):
                for row, metric in enumerate(("Loss", "Accuracy")):
                    axis = axes[row, column]
                    for split in ("Train", "Validation"):
                        points = run["values"][f"{metric}/{split}"]
                        scale = 100 if metric == "Accuracy" else 1
                        axis.plot(
                            [step for step, _ in points],
                            [value * scale for _, value in points], marker="o", label=split,
                        )
                    axis.set_title(f"{name} - {metric}")
                    axis.set_xlabel("Epoch")
                    axis.set_ylabel("Accuracy (%)" if metric == "Accuracy" else "Loss")
                    axis.set_xticks(range(1, args.epochs + 1))
                    axis.grid(alpha=0.3)
                    axis.legend()
                provenance.append({
                    "experiment": name, "run_directory": str(run["directory"]),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    "series": run["values"],
                    "events": [
                        {"path": file, "sha256": hashlib.sha256(Path(file).read_bytes()).hexdigest()}
                        for file in run["files"]
                    ],
                })
            figure.suptitle(f"Existing TensorBoard logs ({args.epochs} epochs)")
            figure.tight_layout()
            with args.output.open("xb") as stream:
                figure.savefig(stream, format="png", dpi=180, bbox_inches="tight")
        finally:
            plt.close(figure)
        with source_path.open("x", encoding="utf-8") as stream:
            json.dump(provenance, stream, ensure_ascii=False, indent=2)
        print(f"训练曲线已导出：{args.output}")
        print(f"来源与原始数值：{source_path}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"无法安全导出曲线：{exc}\n")


if __name__ == "__main__":
    main()
