"""训练入口：必须显式指定实验名称；导入模块或查看帮助不会开始训练。"""
import argparse
import math
import os
import random
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import create_dataloaders
from models.model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
# 历史安全模型永远不能由本入口覆盖，即使指定 --overwrite。
PROTECTED_CHECKPOINTS = {
    (CHECKPOINT_DIR / "baseline_ce_best.pth").resolve(),
    (CHECKPOINT_DIR / "label_smoothing_01_completed.pth").resolve(),
}


def parse_args(argv=None):
    """先校验参数，再创建模型和日志，防止误点击启动训练。"""
    parser = argparse.ArgumentParser(description="宠物分类训练（必须显式指定实验名称）")
    parser.add_argument("--experiment-name", required=True, help="新实验名称，只允许英文字母、数字、_ 和 -")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    # 原 AdamW 未指定此参数，实际默认值为 0.01，现显式保留。
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="0 为 Baseline；0.1 为标签平滑")
    parser.add_argument(
        "--augmentation",
        choices=["basic", "randaugment"],
        default="basic",
        help="训练集增强方式；验证集和测试集始终使用固定预处理",
    )
    parser.add_argument("--seed", type=int, default=42, help="训练随机种子；数据划分仍固定为 42")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名普通模型；安全模型始终受保护")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.experiment_name):
        parser.error("实验名称只能包含英文字母、数字、_ 和 -，且以字母或数字开头。")
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("epochs 和 batch-size 必须为正数；num-workers 不能为负数。")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr 必须是有限正数。")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay 必须是有限非负数。")
    if not 0 <= args.label_smoothing <= 1:
        parser.error("label-smoothing 必须在 [0, 1] 内。")
    if not 0 <= args.seed < 2**32:
        parser.error("seed 必须在 [0, 2**32) 内。")
    return args


def validate_checkpoint_target(
    checkpoint_path,
    overwrite=False,
):
    """检查目标模型路径，防止覆盖已有的重要模型。"""

    # 只有安全模型真实存在时才禁止覆盖。
    # 干净克隆的仓库中没有 checkpoint，因此允许首次复现实验。
    if (
        checkpoint_path.exists()
        and checkpoint_path.resolve() in PROTECTED_CHECKPOINTS
    ):
        raise ValueError(
            f"安全模型禁止覆盖：{checkpoint_path}。"
            "请使用新的 --experiment-name。"
        )

    # 普通模型如果已经存在，默认也不覆盖；
    # 只有显式使用 --overwrite 才允许覆盖。
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(
            f"目标 checkpoint 已存在：{checkpoint_path}。"
            "请更换实验名称；"
            "确需覆盖普通模型时显式加入 --overwrite。"
        )

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch_number,
):
    """
    使用完整训练集训练一个Epoch。

    参数：
        model：
            需要训练的ResNet18模型。

        train_loader：
            训练集DataLoader。
            当前包含161个Batch。

        criterion：
            损失函数。
            当前使用CrossEntropyLoss。

        optimizer：
            优化器。
            当前使用AdamW。

        device：
            训练设备，通常是cuda。

    返回：
        epoch_loss：
            整个Epoch的平均训练损失。

        epoch_accuracy：
            整个Epoch的训练准确率。
    """

    # 将模型切换到训练模式。
    model.train()

    # 用来累计所有图片的损失。
    total_loss = 0.0

    # 用来累计预测正确的图片数量。
    total_correct = 0

    # 用来累计已经处理的图片数量。
    total_samples = 0

    # tqdm为DataLoader添加一个可以实时更新的进度条。
    #
    # enumerate会同时返回：
    # 1. batch_index：当前是第几个Batch
    # 2. images和labels：当前Batch的数据
    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"训练第{epoch_number}个Epoch",
    )

    # 依次取出训练集中的所有Batch。
    for batch_index, (images, labels) in progress_bar:
        # 将当前Batch的图片移动到GPU。
        images = images.to(
            device,
            non_blocking=True,
        )

        # 将正确标签移动到GPU。
        labels = labels.to(
            device,
            non_blocking=True,
        )

        # 清空上一个Batch留下的梯度。
        #
        # 因为PyTorch默认会累加梯度，
        # 所以每个Batch开始前都必须清空。
        optimizer.zero_grad()

        # 前向传播：
        # 模型为每张图片输出37个类别分数。
        logits = model(images)

        # 计算当前Batch的交叉熵损失。
        loss = criterion(logits, labels)

        # 反向传播：
        # 计算每个模型参数的梯度。
        loss.backward()

        # 根据梯度更新模型参数。
        optimizer.step()

        # 当前Batch可能是32张，也可能是最后剩余的24张。
        current_batch_size = labels.size(0)

        # CrossEntropyLoss默认返回当前Batch的平均损失。
        #
        # 这里乘以图片数量，先还原成当前Batch的总损失。
        # 最后再除以整个训练集的图片数，
        # 得到真正按图片数量加权的平均损失。
        total_loss += (
            loss.item() * current_batch_size
        )

        # 每张图片选择分数最高的类别作为预测。
        predictions = logits.argmax(dim=1)

        # 统计当前Batch预测正确了多少张。
        batch_correct = (
            predictions == labels
        ).sum().item()

        total_correct += batch_correct
        total_samples += current_batch_size

        # 计算到目前为止的平均损失。
        current_average_loss = (
            total_loss / total_samples
        )

        # 计算到目前为止的训练准确率。
        current_accuracy = (
            total_correct / total_samples
        )

        # 在进度条右侧实时显示损失和准确率。
        progress_bar.set_postfix(
            loss=f"{current_average_loss:.4f}",
            accuracy=f"{current_accuracy:.2%}",
        )

    # 全部161个Batch处理完成后，
    # 计算整个Epoch的平均损失。
    epoch_loss = total_loss / total_samples

    # 计算整个Epoch的训练准确率。
    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy
def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
    epoch_number,
):
    """
    使用验证集评估当前模型。

    参数：
        model：
            刚刚完成训练的模型。

        val_loader：
            验证集DataLoader。
            当前包含1102张图片、35个Batch。

        criterion：
            与训练阶段相同的交叉熵损失函数。

        device：
            模型和数据所在设备，通常为cuda。

    返回：
        val_loss：
            验证集的平均损失。

        val_accuracy：
            验证集的分类准确率。

    注意：
        验证阶段只检查模型效果，
        不进行反向传播，也不修改参数。
    """

    # 将模型切换到评估模式。
    #
    # 这会改变BatchNorm等层在验证时的行为。
    # 它不会删除模型已经学到的参数。
    model.eval()

    # 累计验证集总损失。
    total_loss = 0.0

    # 累计预测正确的图片数量。
    total_correct = 0

    # 累计已经验证的图片数量。
    total_samples = 0

    # no_grad表示验证期间不计算梯度。
    #
    # 因为验证不需要更新参数，
    # 关闭梯度可以降低显存占用并提升速度。
    with torch.no_grad():
        # 为验证集创建进度条。
        progress_bar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"验证第{epoch_number}个Epoch",
        )

        # 依次读取验证集中的所有Batch。
        for batch_index, (images, labels) in progress_bar:
            # 将验证图片移动到GPU。
            images = images.to(
                device,
                non_blocking=True,
            )

            # 将正确标签移动到GPU。
            labels = labels.to(
                device,
                non_blocking=True,
            )

            # 前向传播，获得37个类别分数。
            logits = model(images)

            # 计算当前Batch的验证损失。
            loss = criterion(logits, labels)

            # 当前Batch的图片数量。
            current_batch_size = labels.size(0)

            # 累加按图片数量加权的损失。
            total_loss += (
                loss.item() * current_batch_size
            )

            # 选出每张图片分数最高的类别。
            predictions = logits.argmax(dim=1)

            # 统计当前Batch预测正确的数量。
            batch_correct = (
                predictions == labels
            ).sum().item()

            total_correct += batch_correct
            total_samples += current_batch_size

            # 计算截至当前Batch的验证平均损失。
            current_average_loss = (
                total_loss / total_samples
            )

            # 计算截至当前Batch的验证准确率。
            current_accuracy = (
                total_correct / total_samples
            )

            # 在进度条上显示累计结果。
            progress_bar.set_postfix(
                loss=f"{current_average_loss:.4f}",
                accuracy=f"{current_accuracy:.2%}",
            )

    # 所有验证Batch处理完成后，
    # 计算整个验证集的平均损失。
    val_loss = total_loss / total_samples

    # 计算整个验证集的准确率。
    val_accuracy = total_correct / total_samples

    return val_loss, val_accuracy
def main(argv=None):
    args = parse_args(argv)
    checkpoint_path = CHECKPOINT_DIR / f"{args.experiment_name}_best.pth"
    # 任何训练、数据读取和 TensorBoard 写入之前，先拒绝危险目标。
    validate_checkpoint_target(checkpoint_path, args.overwrite)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint_path.with_suffix(".lock")
    # x 模式只允许新建，避免两个本脚本进程同时写同一个实验。
    try:
        lock_file = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"实验已被锁定：{lock_path}。确认没有训练进程后再人工处理遗留锁。") from exc

    writer = None
    # 唯一临时文件名，清理时不会误删其他任务留下的文件。
    temp_path = checkpoint_path.with_name(f".{checkpoint_path.stem}.{uuid4().hex}.tmp")
    has_saved = False
    try:
        validate_checkpoint_target(checkpoint_path, args.overwrite)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, val_loader, _ = create_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=False,
            augmentation=args.augmentation,
        )
        model = build_model().to(device)
        # 两组实验共用同一训练流程，唯一的消融变量是此处的平滑系数。
        # 模型直接输出 logits，CrossEntropyLoss 前不添加 Softmax。
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        assert criterion.label_smoothing == args.label_smoothing
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        log_dir = RUNS_DIR / (
            f"{args.experiment_name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        )
        log_dir.mkdir(parents=True, exist_ok=False)
        writer = SummaryWriter(log_dir=str(log_dir))
        writer.add_text("Config", str(vars(args)), 0)
        print("训练设备：", device)
        print("实际配置：", vars(args))
        print("损失函数实际 label_smoothing：", criterion.label_smoothing)
        print("TensorBoard 日志：", log_dir)
        print("最佳模型路径：", checkpoint_path)
        best_val_accuracy = -1.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch,
            )
            val_loss, val_accuracy = validate_one_epoch(
                model, val_loader, criterion, device, epoch,
            )
            writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"Train": train_accuracy, "Validation": val_accuracy}, epoch)
            writer.flush()
            print(
                f"Epoch {epoch}/{args.epochs} | Train Loss {train_loss:.4f}, "
                f"Accuracy {train_accuracy:.2%} | Val Loss {val_loss:.4f}, "
                f"Accuracy {val_accuracy:.2%}"
            )
            # 只以验证集选最佳模型；不在训练中查看测试集指标。
            if val_accuracy > best_val_accuracy:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "learning_rate": args.lr,
                    "num_classes": 37,
                    "model_name": "resnet18",
                    "experiment_name": args.experiment_name,
                    "label_smoothing": args.label_smoothing,
                    "augmentation": args.augmentation,
                    "config": vars(args),
                    "split_seed": 42,
                    "log_dir": str(log_dir.relative_to(PROJECT_ROOT)),
                }
                # 先写临时文件再替换，避免保存中断留下半个 checkpoint。
                # 第一次保存仍检查同名文件，防止启动后被其他程序创建。
                if not has_saved:
                    validate_checkpoint_target(checkpoint_path, args.overwrite)
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, checkpoint_path)
                has_saved = True
                best_val_accuracy = val_accuracy
                print(f"已保存最佳模型：{checkpoint_path}（{best_val_accuracy:.2%}）")
        print(f"训练完成，最佳验证准确率：{best_val_accuracy:.2%}")
    finally:
        if writer is not None:
            writer.close()
        lock_file.close()
        lock_path.unlink(missing_ok=True)
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
