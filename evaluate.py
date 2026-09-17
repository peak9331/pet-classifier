# csv是Python自带的模块，
# 用于将实验指标保存成CSV表格。
import csv

# Path用于安全地处理文件路径。
from pathlib import Path

# torch负责：
# 1. 加载模型权重；
# 2. 将图片移动到GPU；
# 3. 执行模型预测。
import torch
# numpy负责矩阵运算，例如把混淆矩阵转换成百分比。
import numpy as np

# matplotlib负责创建画布并把图片保存到硬盘。
import matplotlib.pyplot as plt

# seaborn负责把二维矩阵绘制成颜色深浅不同的热力图。
import seaborn as sns
# accuracy_score计算Top-1 Accuracy。
# confusion_matrix负责根据真实标签和预测标签生成混淆矩阵。
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

# tqdm用于显示测试进度条。
from tqdm import tqdm

# 导入我们自己编写的数据加载函数。
from data.dataset import create_dataloaders

# 导入我们自己编写的ResNet18创建函数。
from models.model import build_model


# ============================================================
# 1. 路径配置
# ============================================================

# evaluate.py位于项目根目录，
# 所以它的parent就是pet-classifier目录。
PROJECT_ROOT = Path(__file__).resolve().parent

# 模型检查点目录。
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# 评估结果保存目录。
#
# 这里会存放CSV、混淆矩阵和后续Grad-CAM图片。
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 两个需要对比的模型。
#
# 这是一个Python字典：
# 左侧是实验名称，右侧是模型文件路径。
MODEL_PATHS = {
    "baseline_ce": (
        CHECKPOINT_DIR
        / "baseline_ce_best.pth"
    ),
    "label_smoothing_01": (
        CHECKPOINT_DIR
        / "label_smoothing_01_best.pth"
    ),
}
# Oxford-IIIT Pet数据集的37个类别名称。
#
# 列表的位置就是模型使用的类别编号：
# CLASS_NAMES[0]代表类别0，也就是Abyssinian；
# CLASS_NAMES[1]代表类别1，也就是american_bulldog。
#
# 这个顺序必须与数据集标签顺序完全一致，不能按照自己的想法重新排序。
CLASS_NAMES = [
    "Abyssinian",
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "Bengal",
    "Birman",
    "Bombay",
    "boxer",
    "British_Shorthair",
    "chihuahua",
    "Egyptian_Mau",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "Maine_Coon",
    "miniature_pinscher",
    "newfoundland",
    "Persian",
    "pomeranian",
    "pug",
    "Ragdoll",
    "Russian_Blue",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "Siamese",
    "Sphynx",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier",
]

# 自动检查类别数量，防止少写或者多写品种。
assert len(CLASS_NAMES) == 37

# ============================================================
# 2. 加载训练好的模型
# ============================================================

def load_trained_model(
    checkpoint_path,
    device,
):
    """
    从检查点文件中恢复一个训练好的ResNet18。

    参数：
        checkpoint_path：
            模型检查点文件的路径。

            例如：
            checkpoints/baseline_ce_best.pth

        device：
            模型预测时使用的设备。

            可能是：
            cuda
            或者：
            cpu

    返回：
        model：
            已经加载训练参数的ResNet18模型。

        checkpoint：
            检查点中的完整信息字典。
            包含最佳Epoch、验证准确率等。
    """

    # 检查模型文件是否存在。
    #
    # 如果不存在，就立即给出明确错误，
    # 而不是等torch.load时才出现难懂的异常。
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{checkpoint_path}"
        )

    # 从硬盘读取检查点。
    #
    # map_location="cpu"表示先加载到CPU内存。
    # 这样无论保存模型时使用的是GPU还是CPU，
    # 都可以安全读取。
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    # 创建一个新的37分类ResNet18结构。
    #
    # 此时只是创建网络结构，
    # 后面会用检查点参数覆盖它。
    model = build_model()

    # 从检查点中取出训练好的模型参数，
    # 加载到刚刚创建的ResNet18中。
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # 将模型移动到GPU或CPU。
    model = model.to(device)

    # 切换到评估模式。
    #
    # 测试时不需要训练，因此使用eval模式。
    model.eval()

    return model, checkpoint


# ============================================================
# 3. 收集测试集的真实标签和模型预测
# ============================================================

def collect_predictions(
    model,
    test_loader,
    device,
    experiment_name,
):
    """
    使用模型预测整个测试集。

    参数：
        model：
            当前需要评估的模型。

        test_loader：
            测试集DataLoader。
            当前有1103张图片、35个Batch。

        device：
            当前计算设备。

        experiment_name：
            实验名称，只用于显示进度条。

    返回：
        all_true_labels：
            1103张图片的真实标签列表。

        all_predicted_labels：
            模型对1103张图片的预测标签列表。
    """

    # 保存所有真实标签。
    all_true_labels = []

    # 保存所有模型预测。
    all_predicted_labels = []

    # 测试阶段不需要计算梯度。
    #
    # 这样可以：
    # 1. 减少显存占用；
    # 2. 提高预测速度；
    # 3. 防止无意义的梯度记录。
    with torch.no_grad():

        # 依次读取测试集中的35个Batch。
        for images, labels in tqdm(
            test_loader,
            desc=f"测试 {experiment_name}",
        ):
            # 图片需要交给GPU上的模型，
            # 所以必须移动到相同设备。
            images = images.to(
                device,
                non_blocking=True,
            )

            # 前向传播：
            # 输出形状为[当前Batch图片数, 37]。
            logits = model(images)

            # 从每张图片的37个类别分数中，
            # 选择分数最高的类别编号。
            #
            # dim=1表示沿着“类别”这个维度找最大值。
            predicted_labels = logits.argmax(
                dim=1
            )

            # labels原本就在CPU上，
            # 直接转换成普通Python列表。
            all_true_labels.extend(
                labels.tolist()
            )

            # 模型预测当前在GPU上。
            #
            # .cpu()：
            # 将预测结果从GPU显存移动到CPU内存。
            #
            # .tolist()：
            # 将PyTorch张量转换成普通Python列表。
            all_predicted_labels.extend(
                predicted_labels.cpu().tolist()
            )

    return (
        all_true_labels,
        all_predicted_labels,
    )


# ============================================================
# 4. 计算Top-1 Accuracy和Macro-F1
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels,
):
    """
    根据真实标签和预测标签计算分类指标。

    参数：
        true_labels：
            测试集的真实类别列表。

        predicted_labels：
            模型预测类别列表。

    返回：
        top1_accuracy：
            测试集Top-1 Accuracy。

        macro_f1：
            37个类别F1的等权平均值。
    """

    # accuracy_score依次比较：
    #
    # true_labels[i]
    # predicted_labels[i]
    #
    # 相同表示第i张图片预测正确。
    top1_accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    # average="macro"表示：
    #
    # 1. 分别计算37个类别的F1；
    # 2. 将37个F1直接进行等权平均。
    #
    # zero_division=0表示：
    # 如果某个类别完全没有被预测到，
    # 无法计算Precision时，将结果记为0，
    # 避免程序因除以0产生警告。
    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    return top1_accuracy, macro_f1


# ============================================================
# 5. 保存实验对比结果
# ============================================================
# ============================================================
# 5. 绘制混淆矩阵并寻找最容易混淆的类别
# ============================================================

def save_confusion_analysis(
    true_labels,
    predicted_labels,
    class_names,
    experiment_name,
):
    """
    根据测试集的真实标签和预测标签生成混淆矩阵。

    参数：
        true_labels：
            测试集中每张图片的真实类别编号。

        predicted_labels：
            模型对每张图片预测出的类别编号。

        class_names：
            类别编号对应的宠物品种名称。

        experiment_name：
            当前模型的实验名称，用于生成输出文件名。
    """

    # 确保outputs目录存在。
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 创建类别编号：
    # [0, 1, 2, ..., 36]
    #
    # 显式传入labels可以保证最终矩阵一定是37×37。
    class_indices = list(
        range(len(class_names))
    )

    # 生成原始混淆矩阵。
    #
    # matrix[i, j]的含义是：
    # 真实类别为i，但模型预测成类别j的图片数量。
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_indices,
    )

    # 37个真实类别 × 37个预测类别。
    assert matrix.shape == (37, 37)

    # 混淆矩阵里所有数字之和，应当等于测试图片总数1103。
    assert matrix.sum() == len(true_labels)

    # --------------------------------------------------------
    # 1. 将数量矩阵转换成比例矩阵
    # --------------------------------------------------------

    # matrix.sum(axis=1)：
    # 分别计算每一行的总图片数量。
    #
    # keepdims=True：
    # 保留二维形状，使后面的除法能够按行进行。
    row_sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    # 每个格子的数量除以该真实类别的图片总数。
    #
    # 例如某个类别有30张图片，其中27张预测正确：
    # 27 / 30 = 0.9
    #
    # 对角线上就会显示0.9，也就是90%的该类别图片预测正确。
    normalized_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(
            matrix,
            dtype=float,
        ),
        where=row_sums != 0,
    )

    # 为了让坐标轴更容易阅读，
    # 将品种名称中的下划线替换为空格。
    display_names = [
        name.replace("_", " ")
        for name in class_names
    ]

    # --------------------------------------------------------
    # 2. 绘制原始数量混淆矩阵
    # --------------------------------------------------------

    # 创建一张较大的画布。
    # 因为有37个类别，普通尺寸会导致文字重叠。
    plt.figure(figsize=(24, 20))

    # annot=True：
    # 在每个格子中写出图片数量。
    #
    # fmt="d"：
    # 以整数形式显示，例如3，而不是3.0。
    #
    # cmap="Blues"：
    # 数值越大，蓝色越深。
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        annot_kws={"size": 4},
        square=True,
    )

    # 横轴表示模型预测结果。
    plt.xlabel(
        "Predicted breed",
        fontsize=14,
    )

    # 纵轴表示图片真实类别。
    plt.ylabel(
        "True breed",
        fontsize=14,
    )

    plt.title(
        f"Confusion Matrix Counts - {experiment_name}",
        fontsize=16,
    )

    # 调整坐标名称的角度和字号。
    plt.xticks(
        rotation=90,
        fontsize=7,
    )
    plt.yticks(
        rotation=0,
        fontsize=7,
    )

    # 自动调整边距，防止坐标名称被截断。
    plt.tight_layout()

    counts_path = (
        OUTPUT_DIR
        / (
            f"confusion_matrix_"
            f"{experiment_name}_counts.png"
        )
    )

    # dpi=300表示使用较高清晰度保存。
    plt.savefig(
        counts_path,
        dpi=300,
        bbox_inches="tight",
    )

    # 关闭当前画布，释放内存。
    plt.close()

    # --------------------------------------------------------
    # 3. 绘制归一化混淆矩阵
    # --------------------------------------------------------

    plt.figure(figsize=(22, 18))

    # 归一化矩阵中每个数值位于0～1之间。
    #
    # annot=False表示不在每个格子里显示数字，
    # 否则37×37个小数会导致画面非常拥挤。
    sns.heatmap(
        normalized_matrix,
        annot=False,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=display_names,
        yticklabels=display_names,
        square=True,
    )

    plt.xlabel(
        "Predicted breed",
        fontsize=14,
    )
    plt.ylabel(
        "True breed",
        fontsize=14,
    )
    plt.title(
        f"Normalized Confusion Matrix - {experiment_name}",
        fontsize=16,
    )

    plt.xticks(
        rotation=90,
        fontsize=7,
    )
    plt.yticks(
        rotation=0,
        fontsize=7,
    )
    plt.tight_layout()

    normalized_path = (
        OUTPUT_DIR
        / (
            f"confusion_matrix_"
            f"{experiment_name}_normalized.png"
        )
    )

    plt.savefig(
        normalized_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------------
    # 4. 找出最容易混淆的品种组合
    # --------------------------------------------------------

    confusion_pairs = []

    # i只遍历到倒数第二个类别；
    # j从i后面的类别开始遍历。
    #
    # 这样每一对品种只会统计一次，
    # 不会同时出现A-B和B-A两个重复组合。
    for i in range(len(class_names)):
        for j in range(
            i + 1,
            len(class_names),
        ):
            # 真实为i，却预测为j的数量。
            i_to_j = int(matrix[i, j])

            # 真实为j，却预测为i的数量。
            j_to_i = int(matrix[j, i])

            # 将两个方向的错误相加，
            # 得到这一对品种的总混淆次数。
            total_confusions = (
                i_to_j + j_to_i
            )

            # 完全没有混淆的类别组合不用保存。
            if total_confusions == 0:
                continue

            confusion_pairs.append(
                {
                    "breed_a": class_names[i],
                    "breed_b": class_names[j],
                    "a_predicted_as_b": i_to_j,
                    "b_predicted_as_a": j_to_i,
                    "total_confusions": (
                        total_confusions
                    ),
                }
            )

    # 按总混淆次数从大到小排列。
    confusion_pairs.sort(
        key=lambda item: item[
            "total_confusions"
        ],
        reverse=True,
    )

    # 保存排名前10的组合。
    top_confusions = confusion_pairs[:10]

    confusion_csv_path = (
        OUTPUT_DIR
        / (
            f"top_confusions_"
            f"{experiment_name}.csv"
        )
    )

    with confusion_csv_path.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        fieldnames = [
            "breed_a",
            "breed_b",
            "a_predicted_as_b",
            "b_predicted_as_a",
            "total_confusions",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(top_confusions)

    # 在终端中打印最严重的前5对混淆品种。
    print()
    print("===== 最容易混淆的品种组合 =====")

    for rank, item in enumerate(
        top_confusions[:5],
        start=1,
    ):
        print(
            f"{rank}. "
            f"{item['breed_a']} ↔ "
            f"{item['breed_b']}："
            f"共{item['total_confusions']}张"
            f"（A→B：{item['a_predicted_as_b']}，"
            f"B→A：{item['b_predicted_as_a']}）"
        )

    print("数量混淆矩阵已保存：", counts_path)
    print(
        "归一化混淆矩阵已保存：",
        normalized_path,
    )
    print(
        "混淆品种排名已保存：",
        confusion_csv_path,
    )
def save_results_to_csv(results):
    """
    将实验结果保存为CSV表格。

    参数：
        results：
            一个列表，其中每个元素都是一组实验结果字典。
    """

    # 如果outputs目录不存在，就自动创建。
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 最终保存路径：
    # outputs/experiment_results.csv
    csv_path = (
        OUTPUT_DIR
        / "experiment_results.csv"
    )

    # newline=""避免Windows写CSV时出现空行。
    #
    # encoding="utf-8-sig"可以让Excel
    # 更好地识别中文和UTF-8编码。
    with csv_path.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        # 定义CSV表格的列顺序。
        fieldnames = [
            "experiment",
            "best_epoch",
            "val_accuracy",
            "test_top1_accuracy",
            "test_macro_f1",
        ]

        # 创建字典形式的CSV写入器。
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        # 写入第一行表头。
        writer.writeheader()

        # 写入所有实验结果。
        writer.writerows(results)

    print("实验结果已保存：", csv_path)


# ============================================================
# 6. 主程序
# ============================================================

def main():
    """
    依次评估Baseline和标签平滑模型。
    """

    # 如果CUDA可用就使用GPU。
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("===== 测试集对比评估 =====")
    print("评估设备：", device)

    # 创建训练、验证和测试DataLoader。
    #
    # 前两个下划线表示：
    # 当前不使用train_loader和val_loader。
    #
    # 我们只取第三个返回值test_loader。
    _, _, test_loader = create_dataloaders()

    # 保存两个模型的最终结果。
    results = []

    # MODEL_PATHS.items()会依次返回：
    #
    # experiment_name：
    #     例如baseline_ce
    #
    # checkpoint_path：
    #     对应的模型文件路径
    for experiment_name, checkpoint_path in (
        MODEL_PATHS.items()
    ):
        print()
        print(
            f"===== 评估 {experiment_name} ====="
        )
        print("模型路径：", checkpoint_path)

        # 加载当前实验的最佳模型。
        model, checkpoint = load_trained_model(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        # 让模型预测全部1103张测试图片。
        true_labels, predicted_labels = (
            collect_predictions(
                model=model,
                test_loader=test_loader,
                device=device,
                experiment_name=experiment_name,
            )
        )

        # 自动检查真实标签和预测数量是否一致。
        assert (
            len(true_labels)
            == len(predicted_labels)
        )

        # 当前测试集应当有1103张图片。
        print(
            "测试图片数量：",
            len(true_labels),
        )

        # 计算Top-1 Accuracy和Macro-F1。
        top1_accuracy, macro_f1 = (
            calculate_metrics(
                true_labels=true_labels,
                predicted_labels=predicted_labels,
            )
        )
        # 只为当前表现更好的Label Smoothing模型
        # 生成混淆矩阵。
        #
        # Baseline仍然会正常计算Accuracy和Macro-F1，
        # 但不会重复生成另一套混淆矩阵。
        if experiment_name == "label_smoothing_01":
            save_confusion_analysis(
                true_labels=true_labels,
                predicted_labels=predicted_labels,
                class_names=CLASS_NAMES,
                experiment_name=experiment_name,
            )
        # 从检查点读取验证阶段信息。
        best_epoch = checkpoint["epoch"]
        val_accuracy = checkpoint[
            "val_accuracy"
        ]

        print("最佳Epoch：", best_epoch)
        print(
            "验证准确率：",
            f"{val_accuracy:.2%}",
        )
        print(
            "测试集Top-1 Accuracy：",
            f"{top1_accuracy:.2%}",
        )
        print(
            "测试集Macro-F1：",
            f"{macro_f1:.4f}",
        )

        # 将当前实验结果保存到列表中。
        results.append(
            {
                "experiment": experiment_name,
                "best_epoch": best_epoch,

                # CSV中保存原始小数，
                # 方便以后继续计算和绘图。
                "val_accuracy": val_accuracy,
                "test_top1_accuracy": (
                    top1_accuracy
                ),
                "test_macro_f1": macro_f1,
            }
        )

        # 删除当前模型对象，
        # 为下一个模型释放Python引用。
        del model

        # 如果使用GPU，尝试清理当前无用缓存。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 将两组实验结果保存为CSV。
    save_results_to_csv(results)

    print()
    print("===== 两组实验测试完成 =====")


if __name__ == "__main__":
    main()