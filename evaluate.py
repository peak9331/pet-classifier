"""独立评估两份历史模型，计算 Top-1、Top-5、Macro-F1。"""
import csv
import io
import math
import os
from pathlib import Path

import torch
from tqdm import tqdm

from data.dataset import create_dataloaders
from utils.checkpoints import CLASS_NAMES, load_trained_model
from utils.metrics import calculate_metrics, save_confusion_analysis

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATHS = {
    "baseline_ce": CHECKPOINT_DIR / "baseline_ce_best.pth",
    # 使用已确认的完整备份，避免误读被中途训练覆盖的同名 best 文件。
    "label_smoothing_01": CHECKPOINT_DIR / "label_smoothing_01_best.pth",
}


def collect_predictions(model, test_loader, device, experiment_name):
    """一次前向传播同时取得 Top-1 和 Top-5，不改变模型参数。"""
    model.eval()
    true_labels, predicted_labels, top5_labels = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"测试 {experiment_name}"):
            logits = model(images.to(device, non_blocking=True))
            true_labels.extend(labels.tolist())
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())
            # logits 排序即可，不需要 Softmax；Top-5 每张图片保存五个类别编号。
            top5_labels.extend(logits.topk(k=5, dim=1).indices.cpu().tolist())
    return true_labels, predicted_labels, top5_labels


def save_results_to_csv(results):
    """为历史结果增补 Top-5；旧字段发生冲突时拒绝覆盖，保留其他实验行。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "reproduced_experiment_results.csv"
    fieldnames = [
        "experiment", "best_epoch", "val_accuracy", "test_top1_accuracy",
        "test_top5_accuracy", "test_macro_f1",
    ]
    old_bytes = csv_path.read_bytes() if csv_path.exists() else None
    rows = []
    if old_bytes is not None:
        reader = csv.DictReader(io.StringIO(old_bytes.decode("utf-8-sig")))
        fieldnames += [key for key in (reader.fieldnames or []) if key not in fieldnames]
        rows = list(reader)
    by_experiment = {row["experiment"]: row for row in rows}
    for result in results:
        existing = by_experiment.get(result["experiment"])
        if existing is None:
            rows.append(result)
            continue
        for key, value in result.items():
            if key == "experiment":
                continue
            if existing.get(key):
                if not math.isclose(float(existing[key]), float(value), rel_tol=0, abs_tol=1e-10):
                    raise ValueError(f"复评结果与已有记录不一致：{result['experiment']} / {key}；原 CSV 保留。")
            else:
                existing[key] = value
    # 只有两个模型都成功评估且与历史结果一致，才更新总表。
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    new_bytes = output.getvalue().encode("utf-8-sig")
    if old_bytes == new_bytes:
        print("实验结果没有变化：", csv_path)
        return
    backup = OUTPUT_DIR / "experiment_results_before_day3.csv"
    if old_bytes is not None and not backup.exists():
        with backup.open("xb") as stream:
            stream.write(old_bytes)
    temp_path = csv_path.with_suffix(".csv.tmp")
    temp_created = False
    try:
        with temp_path.open("xb") as stream:
            temp_created = True
            stream.write(new_bytes)
        os.replace(temp_path, csv_path)
    finally:
        # 只删除本次创建的临时文件；原 CSV 和备份不会删除。
        if temp_created and temp_path.exists():
            temp_path.unlink()
    print("实验结果已保存：", csv_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("评估设备：", device)
    # 数据缺失时停止，评估过程中不下载、改写 datasets。
    _, _, test_loader = create_dataloaders(download=False)
    results = []
    for experiment_name, checkpoint_path in MODEL_PATHS.items():
        print("模型路径：", checkpoint_path)
        model, checkpoint = load_trained_model(checkpoint_path, device)
        true_labels, predicted_labels, top5_labels = collect_predictions(
            model, test_loader, device, experiment_name,
        )
        if len(true_labels) != len(test_loader.dataset):
            raise ValueError("测试集没有被完整评估。")
        metrics = calculate_metrics(true_labels, predicted_labels, top5_labels, len(CLASS_NAMES))
        result = {
            "experiment": experiment_name,
            "best_epoch": checkpoint["epoch"],
            "val_accuracy": checkpoint["val_accuracy"],
            **metrics,
        }
        results.append(result)
        print(f"最佳 Epoch：{result['best_epoch']}；验证准确率：{result['val_accuracy']:.2%}")
        print(f"测试样本：{len(true_labels)}")
        print(f"Top-1：{metrics['test_top1_accuracy']:.2%}；Top-5：{metrics['test_top5_accuracy']:.2%}；Macro-F1：{metrics['test_macro_f1']:.4f}")
        if experiment_name == "label_smoothing_01":
            save_confusion_analysis(
                true_labels, predicted_labels, CLASS_NAMES, experiment_name, OUTPUT_DIR,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_results_to_csv(results)


if __name__ == "__main__":
    main()
