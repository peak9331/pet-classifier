"""统一评估三组宠物分类模型，计算 Top-1、Top-5 和 Macro-F1。"""
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
    "baseline_ce": (
        CHECKPOINT_DIR / "baseline_ce_best.pth"
    ),
    "label_smoothing_01": (
        CHECKPOINT_DIR / "label_smoothing_01_best.pth"
    ),
    "randaugment_n2_m9": (
        CHECKPOINT_DIR / "randaugment_n2_m9_best.pth"
    ),
}
REPRODUCED_RESULTS_PATH = (
    OUTPUT_DIR / "reproduced_experiment_results.csv"
)

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
    """
    保存当前电脑重新训练和评估得到的结果。

    该文件与仓库中的参考结果分开，
    避免不同硬件产生的小幅差异导致评估失败。
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "experiment",
        "augmentation",
        "label_smoothing",
        "best_epoch",
        "val_accuracy",
        "test_top1_accuracy",
        "test_top5_accuracy",
        "test_macro_f1",
    ]

    with REPRODUCED_RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        "复现实验结果已保存：",
        REPRODUCED_RESULTS_PATH,
    )

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

            # 旧 checkpoint 可能没有 augmentation 字段。
            # 前两组旧实验实际使用的是 basic，因此缺失时按 basic 处理。
            "augmentation": checkpoint.get(
                "augmentation",
                "basic",
            ),

            "label_smoothing": checkpoint.get(
                "label_smoothing",
                0.0,
            ),

            "best_epoch": checkpoint["epoch"],
            "val_accuracy": checkpoint["val_accuracy"],
            **metrics,
        }
        results.append(result)
        print(
            "实验配置："
            f"augmentation={result['augmentation']}，"
            f"label_smoothing={result['label_smoothing']}"
        )
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
