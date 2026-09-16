from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


# ============================================================
# 1. 项目基础配置
# ============================================================

# __file__ 表示当前文件：
# D:\projects\pet-classifier\data\dataset.py
#
# 第一个 parent 得到 data 文件夹，
# 第二个 parent 得到项目根目录 pet-classifier。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 原始图片将存放在项目根目录下的 datasets 文件夹。
# Path 对象使用 / 拼接路径，可以兼容 Windows、Linux 和 macOS。
DATA_ROOT = PROJECT_ROOT / "datasets"

# Oxford-IIIT Pet 一共有37个宠物品种。
NUM_CLASSES = 37

# 固定随机种子，使每次运行都得到相同的数据划分。
# 42没有特殊含义，只是机器学习项目中常用的固定数字。
RANDOM_SEED = 42

# 每次向模型提供32张图片。
# RTX 4060 Laptop 的8GB显存通常可以从32开始尝试。
BATCH_SIZE = 32


# ============================================================
# 2. 下载并加载原始数据
# ============================================================

def load_raw_datasets():
    """
    下载并加载 Oxford-IIIT Pet 的两个官方数据部分。

    当前函数只负责读取原始数据，不负责：
    1. 重新划分训练集、验证集、测试集；
    2. 调整图片大小；
    3. 把图片转换为张量。
    """

    # 官方的 trainval 部分包含3680张图片。
    trainval_dataset = OxfordIIITPet(
        root=DATA_ROOT,
        split="trainval",

        # category 表示分类任务的宠物品种标签。
        # 标签取值为0～36，共37种。
        target_types="category",

        # 如果数据不存在就自动下载；
        # 如果已经下载完成，就直接读取，不会重复下载。
        download=True,
    )

    # 官方的 test 部分包含3669张图片。
    test_dataset = OxfordIIITPet(
        root=DATA_ROOT,
        split="test",
        target_types="category",
        download=True,
    )

    return trainval_dataset, test_dataset


# ============================================================
# 3. 合并官方数据并取得所有标签
# ============================================================

def combine_datasets(trainval_dataset, official_test_dataset):
    """
    将官方 trainval 和 test 合并成一个完整数据集。

    为什么要合并？
    因为本项目要求重新按照70% / 15% / 15%分层划分，
    所以暂时将7349张图片看成一个完整的数据池。
    """

    # ConcatDataset 不会复制图片，
    # 它只是把两个数据集组织成一个可以连续索引的数据集。
    full_dataset = ConcatDataset(
        [trainval_dataset, official_test_dataset]
    )

    # _labels 中保存了每张图片的类别编号。
    #
    # 注意：
    # 名称以下划线开头，说明它属于 torchvision 的内部属性。
    # 当前固定版本可以使用，但将来升级 torchvision 时需要重新检查。
    trainval_targets = np.asarray(trainval_dataset._labels)
    test_targets = np.asarray(official_test_dataset._labels)

    # 将两个标签数组连接起来。
    # targets[i] 就是 full_dataset[i] 对应的类别。
    targets = np.concatenate(
        [trainval_targets, test_targets]
    )

    # 数据和标签的数量必须相同，否则后续划分会发生错位。
    assert len(full_dataset) == len(targets)

    return full_dataset, targets


# ============================================================
# 4. 按70% / 15% / 15%进行分层划分
# ============================================================

def stratified_split(targets):
    """
    将完整数据分成：
    - 70% 训练集
    - 15% 验证集
    - 15% 测试集

    stratify 参数会尽量维持37个类别在各部分中的比例。
    """

    # 创建所有样本的编号：
    # [0, 1, 2, ..., 7348]
    all_indices = np.arange(len(targets))

    # 第一次划分：
    # 70%进入训练集，30%暂存在 temp_indices 中。
    train_indices, temp_indices = train_test_split(
        all_indices,

        # 暂时取出30%，后面再平均分成验证集和测试集。
        test_size=0.30,

        # 固定随机种子，保证每次划分一致。
        random_state=RANDOM_SEED,

        # 按标签分层抽样，而不是完全随机抽样。
        stratify=targets,
    )

    # 第二次划分：
    # 把刚才的30%平均分成15%验证集和15%测试集。
    val_indices, test_indices = train_test_split(
        temp_indices,

        # 从临时数据中取一半作为测试集。
        # 原始数据的30% × 50% = 15%。
        test_size=0.50,

        random_state=RANDOM_SEED,

        # 这里只能传入临时数据对应的标签。
        stratify=targets[temp_indices],
    )

    return train_indices, val_indices, test_indices


# ============================================================
# 5. 验证数据划分是否正确
# ============================================================

def verify_split(
    train_indices,
    val_indices,
    test_indices,
    targets,
):
    """
    检查三个数据集是否存在重复、遗漏或类别缺失。

    科研中不能只相信代码“应该正确”，
    还要通过断言和统计结果验证它确实正确。
    """

    # 转换为集合，方便判断是否有相同下标。
    train_set = set(train_indices.tolist())
    val_set = set(val_indices.tolist())
    test_set = set(test_indices.tolist())

    # isdisjoint 表示两个集合没有共同元素。
    # 如果断言失败，说明同一张图片被分到了两个数据集。
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    # 三个数据集的数量之和必须等于完整数据量。
    assert (
        len(train_indices)
        + len(val_indices)
        + len(test_indices)
        == len(targets)
    )

    total = len(targets)

    print("===== 数据划分结果 =====")
    print(f"总数据：{total}")

    print(
        f"训练集：{len(train_indices)}，"
        f"占比：{len(train_indices) / total:.2%}"
    )

    print(
        f"验证集：{len(val_indices)}，"
        f"占比：{len(val_indices) / total:.2%}"
    )

    print(
        f"测试集：{len(test_indices)}，"
        f"占比：{len(test_indices) / total:.2%}"
    )

    # 分别检查训练、验证和测试集中的类别数量。
    for name, indices in [
        ("训练集", train_indices),
        ("验证集", val_indices),
        ("测试集", test_indices),
    ]:
        # bincount 统计每个类别分别有多少张图片。
        # minlength=37 保证结果中始终包含37个类别的位置。
        class_counts = np.bincount(
            targets[indices],
            minlength=NUM_CLASSES,
        )

        # 非零位置数量应该为37，表示没有类别缺失。
        existing_class_count = np.count_nonzero(class_counts)

        print(
            f"{name}："
            f"{existing_class_count}个类别，"
            f"单类最少{class_counts.min()}张，"
            f"单类最多{class_counts.max()}张"
        )

        # 如果不是37个类别，就立即停止程序。
        assert existing_class_count == NUM_CLASSES


# ============================================================
# 6. 定义图片预处理
# ============================================================

# ResNet18 的预训练权重是在 ImageNet 数据集上训练得到的，
# 因此输入图片应使用 ImageNet 的均值和标准差进行标准化。
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# 训练集预处理。
# 训练集允许加入随机操作，增加图片变化，减少模型死记硬背。
train_transform = transforms.Compose(
    [
        # 将图片较短的一边缩放到256。
        transforms.Resize(256),

        # 随机裁剪一个224×224的区域。
        transforms.RandomCrop(224),

        # 以50%的概率水平翻转图片。
        transforms.RandomHorizontalFlip(),

        # 将PIL图片转换为PyTorch张量。
        #
        # 转换前通常是：
        # 高 × 宽 × 通道，像素范围0～255。
        #
        # 转换后是：
        # 通道 × 高 × 宽，像素范围0～1。
        transforms.ToTensor(),

        # 使用ImageNet参数标准化。
        # Normalize必须放在ToTensor之后。
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


# 验证集和测试集预处理。
# 这里不能使用随机裁剪或随机翻转，
# 否则同一张图片每次评估结果可能不同。
eval_transform = transforms.Compose(
    [
        transforms.Resize(256),

        # 从图片中心裁剪224×224区域。
        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


# ============================================================
# 7. 为数据子集应用不同的预处理
# ============================================================

class TransformedSubset(Dataset):
    """
    从完整数据集中选择指定下标，
    并在读取图片时应用对应的预处理。

    训练集使用随机增强，
    验证集和测试集使用固定预处理。
    """

    def __init__(
        self,
        dataset,
        indices,
        transform,
    ):
        # 保存完整原始数据集。
        self.dataset = dataset

        # 保存当前子集包含的样本下标。
        self.indices = indices

        # 保存当前子集使用的图片预处理。
        self.transform = transform

    def __len__(self):
        # 告诉PyTorch当前子集有多少条数据。
        return len(self.indices)

    def __getitem__(self, item):
        # item 是当前子集中的位置。
        # 先将它转换成完整数据集中的真实下标。
        original_index = int(self.indices[item])

        # 从完整数据集中取得原始PIL图片和类别标签。
        image, label = self.dataset[original_index]

        # 如果设置了预处理，就对图片进行转换。
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# 8. 创建DataLoader
# ============================================================

def create_dataloaders(batch_size=BATCH_SIZE):
    """
    完成完整的数据流水线：

    下载数据
        ↓
    合并数据
        ↓
    分层划分
        ↓
    应用预处理
        ↓
    创建DataLoader
    """

    # 下载并读取两个官方部分。
    trainval_dataset, official_test_dataset = (
        load_raw_datasets()
    )

    # 合并成完整数据集，并取得所有标签。
    full_dataset, targets = combine_datasets(
        trainval_dataset,
        official_test_dataset,
    )

    # 进行70% / 15% / 15%分层划分。
    train_indices, val_indices, test_indices = (
        stratified_split(targets)
    )

    # 检查划分是否存在重复、遗漏或类别缺失。
    verify_split(
        train_indices,
        val_indices,
        test_indices,
        targets,
    )

    # 训练集使用带有随机增强的预处理。
    train_dataset = TransformedSubset(
        dataset=full_dataset,
        indices=train_indices,
        transform=train_transform,
    )

    # 验证集使用固定预处理。
    val_dataset = TransformedSubset(
        dataset=full_dataset,
        indices=val_indices,
        transform=eval_transform,
    )

    # 测试集同样使用固定预处理。
    test_dataset = TransformedSubset(
        dataset=full_dataset,
        indices=test_indices,
        transform=eval_transform,
    )

    # 训练集需要shuffle=True。
    # 每轮训练开始前打乱顺序，避免模型记住数据排列。
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,

        # Windows下先使用0，表示由主进程加载数据。
        # 这样最稳定。跑通后再尝试调整到2或4。
        num_workers=0,

        # 使用GPU时，锁页内存可以加快CPU到GPU的数据传输。
        pin_memory=torch.cuda.is_available(),
    )

    # 验证集不能打乱，因为不参与参数训练。
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # 测试集同样不需要打乱。
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 9. 运行数据流水线检查
# ============================================================

def main():
    # 创建三个DataLoader。
    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    # 从训练集中取出第一个批次。
    images, labels = next(iter(train_loader))

    print()
    print("===== Batch检查 =====")

    # 正确结果应为：
    # torch.Size([32, 3, 224, 224])
    #
    # 32：一个批次有32张图片
    # 3：RGB三个颜色通道
    # 224、224：图片高度和宽度
    print("图片形状：", images.shape)

    # 正确结果应为：
    # torch.Size([32])
    #
    # 表示32张图片分别有一个标签。
    print("标签形状：", labels.shape)

    # 图片经过ToTensor后通常应为torch.float32。
    print("图片数据类型：", images.dtype)

    # 分类标签通常应为torch.int64。
    # CrossEntropyLoss要求标签使用整数类别编号。
    print("标签数据类型：", labels.dtype)

    # 查看当前批次中标签的最小值和最大值。
    # 因为这里只取一个随机批次，不一定包含0和36。
    print("本批次标签最小值：", labels.min().item())
    print("本批次标签最大值：", labels.max().item())

    # len(DataLoader)表示一个Epoch包含多少个批次。
    print("训练集Batch数量：", len(train_loader))
    print("验证集Batch数量：", len(val_loader))
    print("测试集Batch数量：", len(test_loader))


# 只有直接运行当前文件时，才执行main函数。
# 当train.py导入create_dataloaders时，不会自动运行这些检查。
if __name__ == "__main__":
    main()