"""分类指标与混淆矩阵工具；不依赖任何根目录运行入口。"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns


def calculate_metrics(true_labels, predicted_labels, top5_labels, num_classes=37):
    """Top-1/Top-5 返回 0～1 比例；Macro-F1 对所有类别等权平均。"""
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    top5_labels = np.asarray(top5_labels)
    if true_labels.ndim != 1 or not len(true_labels):
        raise ValueError("真实标签必须是非空的一维数组。")
    if predicted_labels.shape != true_labels.shape:
        raise ValueError("预测标签数量与真实标签数量不一致。")
    if top5_labels.shape != (len(true_labels), min(5, num_classes)):
        raise ValueError("Top-5 候选形状应为 [样本数, min(5, 类别数)]。")
    # 一个样本只要命中五个候选中的任意一个，就记为 Top-5 正确。
    top5_accuracy = np.any(top5_labels == true_labels[:, None], axis=1).mean()
    return {
        "test_top1_accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "test_top5_accuracy": float(top5_accuracy),
        "test_macro_f1": float(f1_score(
            true_labels, predicted_labels, labels=list(range(num_classes)),
            average="macro", zero_division=0,
        )),
    }


def calculate_confusion_matrix(true_labels, predicted_labels, num_classes=37):
    """行是真实类别，列是预测类别；显式标签确保缺失类别仍有对应行列。"""
    matrix = confusion_matrix(true_labels, predicted_labels, labels=range(num_classes))
    if int(matrix.sum()) != len(true_labels):
        raise ValueError("混淆矩阵样本数异常，请检查类别编号。")
    return matrix


def normalize_confusion_matrix(matrix):
    """按真实类别逐行归一化；空行保持 0，避免除零。"""
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)


def _plot_matrix(matrix, class_names, output_path, title, normalized):
    output_path = Path(output_path)
    # 保留历史图片；需要重绘时传入新的输出路径。
    if output_path.exists():
        print(f"保留已有图片：{output_path}")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(22, 18) if normalized else (24, 20))
    display_names = [name.replace("_", " ") for name in class_names]
    try:
        sns.heatmap(
            matrix, annot=not normalized, fmt=".2f" if normalized else "d",
            cmap="Blues", vmin=0, vmax=1 if normalized else None,
            xticklabels=display_names, yticklabels=display_names,
            annot_kws={"size": 4}, square=True, ax=ax,
        )
        ax.set(xlabel="Predicted breed", ylabel="True breed", title=title)
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.tick_params(axis="y", labelrotation=0, labelsize=7)
        figure.tight_layout()
        # x 模式再次保护同名文件，避免静默覆盖。
        with output_path.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output_path


def plot_confusion_matrix_counts(matrix, class_names, output_path, experiment_name):
    return _plot_matrix(matrix, class_names, output_path, f"Confusion Matrix Counts - {experiment_name}", False)


def plot_confusion_matrix_normalized(matrix, class_names, output_path, experiment_name):
    return _plot_matrix(normalize_confusion_matrix(matrix), class_names, output_path, f"Normalized Confusion Matrix - {experiment_name}", True)


def top_confusion_pairs(matrix, class_names, top_n=10):
    """每对类别只统计一次，A→B 与 B→A 相加；不包含正确分类的对角线。"""
    pairs = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            a_to_b, b_to_a = int(matrix[i, j]), int(matrix[j, i])
            if a_to_b + b_to_a:
                pairs.append({
                    "breed_a": class_names[i], "breed_b": class_names[j],
                    "a_predicted_as_b": a_to_b, "b_predicted_as_a": b_to_a,
                    "total_confusions": a_to_b + b_to_a,
                })
    pairs.sort(key=lambda item: item["total_confusions"], reverse=True)
    return pairs[:top_n]


def save_confusion_analysis(true_labels, predicted_labels, class_names, experiment_name, output_dir):
    """计算分析结果；历史图片和排名 CSV 已存在时保留原文件。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = calculate_confusion_matrix(true_labels, predicted_labels, len(class_names))
    pairs = top_confusion_pairs(matrix, class_names)
    plot_confusion_matrix_counts(
        matrix, class_names, output_dir / f"confusion_matrix_{experiment_name}_counts.png", experiment_name,
    )
    plot_confusion_matrix_normalized(
        matrix, class_names, output_dir / f"confusion_matrix_{experiment_name}_normalized.png", experiment_name,
    )
    csv_path = output_dir / f"top_confusions_{experiment_name}.csv"
    if not csv_path.exists():
        with csv_path.open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "breed_a", "breed_b", "a_predicted_as_b", "b_predicted_as_a", "total_confusions",
            ])
            writer.writeheader()
            writer.writerows(pairs)
    else:
        print(f"保留已有排名：{csv_path}")
    for pair in pairs[:5]:
        print(f"{pair['breed_a']} ↔ {pair['breed_b']}：{pair['total_confusions']} 次")
    return matrix, pairs
