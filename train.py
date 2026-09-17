# datetime用于为每次实验生成不同的日志名称。
from datetime import datetime
# Path用于处理日志和模型保存路径。$
from pathlib import Path
# SummaryWriter负责向TensorBoard写入训练数据。
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
# tqdm可以显示训练进度条。
from tqdm import tqdm
# 导入刚才完成的数据加载函数。
from data.dataset import create_dataloaders

# 导入刚才完成的模型创建函数。
from models.model import build_model


# 固定随机种子，尽量让实验结果可以复现。
RANDOM_SEED = 42

# 学习率控制模型每次更新参数的幅度。
#
# 1e-4 等于 0.0001。
# 对预训练模型进行微调时，通常使用比较小的学习率。
LEARNING_RATE = 1e-4
# 当前Baseline正式训练10个Epoch。
NUM_EPOCHS = 10
# ============================================================
# 当前实验配置
# ============================================================

# 每组实验必须使用清晰且唯一的名称。
#
# 这个名称会用于：
# 1. TensorBoard日志文件夹；
# 2. 最佳模型文件名；
# 3. 检查点中的实验信息。
EXPERIMENT_NAME = "label_smoothing_01"

# 标签平滑系数。
#
# 0.0表示普通交叉熵；
# 0.1表示拿出10%的权重与均匀分布混合。
LABEL_SMOOTHING = 0.1

# train.py就在项目根目录中，
# 因此它的parent就是pet-classifier。
PROJECT_ROOT = Path(__file__).resolve().parent

# 模型权重保存目录：
# pet-classifier/checkpoints
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# TensorBoard日志根目录：
# pet-classifier/runs
RUNS_DIR = PROJECT_ROOT / "runs"

# 使用实验名称生成独立的模型文件名。
#
# 最终路径为：
# checkpoints/label_smoothing_01_best.pth
BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / f"{EXPERIMENT_NAME}_best.pth"
)

def train_one_batch():
    """
    使用一个Batch完成一次最小训练。

    这个函数不是完整训练循环，
    它只验证以下组件能否正确配合：

    1. DataLoader
    2. ResNet18
    3. GPU
    4. CrossEntropyLoss
    5. AdamW
    6. 反向传播
    """

    # 固定PyTorch的随机种子。
    torch.manual_seed(RANDOM_SEED)

    # 如果CUDA可用就使用GPU，否则使用CPU。
    #
    # device最终会是：
    # torch.device("cuda")
    # 或者：
    # torch.device("cpu")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("===== 创建训练组件 =====")
    print("训练设备：", device)

    # 创建训练集、验证集和测试集的DataLoader。
    #
    # 当前只使用train_loader，
    # 但先接收全部三个返回值，方便以后扩展完整训练循环。
    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    # 创建37分类的ResNet18模型。
    model = build_model()

    # 将模型中的所有参数移动到GPU。
    #
    # 如果不执行这一行，模型仍然在CPU上。
    model = model.to(device)

    # 创建交叉熵损失函数。
    #
    # CrossEntropyLoss用于多分类问题。
    #
    # 它接收：
    # 1. 模型输出的logits，形状为[32, 37]
    # 2. 正确标签，形状为[32]
    #
    # 注意：
    # 使用CrossEntropyLoss时，模型后面不要手动添加Softmax。
    # 创建带标签平滑的交叉熵损失。
    #
    # 当前LABEL_SMOOTHING为0.1。
    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
    )

    # 创建AdamW优化器。
    #
    # model.parameters()表示模型中所有需要学习的参数。
    #
    # lr表示learning rate，也就是学习率。
    #
    # 优化器的任务是：
    # 根据参数的梯度，实际修改模型参数。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # 从训练集中取得第一个Batch。
    #
    # images形状为：
    # [32, 3, 224, 224]
    #
    # labels形状为：
    # [32]
    images, labels = next(iter(train_loader))

    # 将图片和标签从CPU内存移动到GPU显存。
    #
    # 模型和数据必须位于同一个设备上。
    # 不能让模型在GPU、图片却在CPU。
    images = images.to(
        device,
        non_blocking=True,
    )

    labels = labels.to(
        device,
        non_blocking=True,
    )

    # 切换到训练模式。
    #
    # model.train()不会立即开始训练，
    # 它只是告诉模型：
    # “接下来的操作属于训练阶段。”
    model.train()

    # 保存更新前最后一层的参数。
    #
    # detach()表示这里只读取参数，不加入梯度计算。
    # clone()表示复制一份，防止后面参数变化影响这份记录。
    weight_before = model.fc.weight.detach().clone()

    # 清空上一次计算留下的梯度。
    #
    # PyTorch默认会累加梯度。
    # 如果不清空，本次梯度会和之前的梯度加在一起。
    optimizer.zero_grad()

    # ========================================================
    # 第一步：前向传播
    # ========================================================

    # 将32张图片送入模型。
    logits = model(images)

    # logits形状应为：
    # [32, 37]
    #
    # 每一行代表一张图片，
    # 每一列代表一个宠物类别的预测分数。
    print()
    print("===== 前向传播 =====")
    print("输入图片形状：", images.shape)
    print("正确标签形状：", labels.shape)
    print("模型输出形状：", logits.shape)

    # ========================================================
    # 第二步：计算损失
    # ========================================================

    # 将模型预测与正确标签进行比较。
    #
    # loss是一个数值，用于表示模型当前错得有多严重。
    # 通常loss越小越好。
    loss = criterion(logits, labels)

    print("当前Batch损失：", loss.item())

    # ========================================================
    # 第三步：反向传播
    # ========================================================

    # 根据loss计算每个模型参数应该如何改变。
    #
    # backward()只计算梯度，
    # 还没有真正修改模型参数。
    loss.backward()

    # ========================================================
    # 第四步：更新参数
    # ========================================================

    # 优化器根据刚才计算出的梯度修改模型参数。
    optimizer.step()

    # 保存更新后的最后一层参数。
    weight_after = model.fc.weight.detach().clone()

    # 计算参数更新前后的平均变化量。
    #
    # 只要结果大于0，就说明优化器确实修改了参数。
    average_weight_change = (
        weight_after - weight_before
    ).abs().mean().item()

    # ========================================================
    # 第五步：计算当前Batch准确率
    # ========================================================

    # logits中每一行有37个类别分数。
    #
    # argmax(dim=1)会找到每张图片分数最高的类别编号。
    predictions = logits.argmax(dim=1)

    # 判断预测类别是否等于正确标签。
    correct_count = (
        predictions == labels
    ).sum().item()

    # 当前Batch一共有多少张图片。
    batch_size = labels.size(0)

    # 计算当前Batch的准确率。
    batch_accuracy = correct_count / batch_size

    print()
    print("===== 参数更新检查 =====")
    print("正确预测数量：", correct_count)
    print("当前Batch图片数量：", batch_size)
    print(
        f"当前Batch准确率："
        f"{batch_accuracy:.2%}"
    )
    print(
        "最后一层参数平均变化量：",
        average_weight_change,
    )

    # ========================================================
    # 第六步：自动检查关键结果
    # ========================================================

    # 32张图片必须分别输出37个类别分数。
    assert logits.shape == (
        batch_size,
        37,
    )

    # 损失必须是一个正常的有限数值。
    #
    # 如果出现NaN或无穷大，
    # 通常说明数据、学习率或计算过程存在问题。
    assert torch.isfinite(loss)

    # 参数变化量必须大于0。
    # 否则说明优化器没有成功更新模型。
    assert average_weight_change > 0

    print()
    print("一次Batch训练成功！")
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
def main():
    """
    完成正式Baseline训练：

    1. 创建数据、模型、损失函数和优化器；
    2. 循环训练10个Epoch；
    3. 每个Epoch结束后进行验证；
    4. 将结果写入TensorBoard；
    5. 自动保存验证准确率最高的模型。
    """

    # 固定CPU上的PyTorch随机种子。
    torch.manual_seed(RANDOM_SEED)

    # 如果使用GPU，也固定GPU随机种子。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # 优先使用GPU。
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("实验名称：", EXPERIMENT_NAME)
    print("标签平滑系数：", LABEL_SMOOTHING)
    print("===== Baseline正式训练 =====")
    print("训练设备：", device)
    print("训练轮数：", NUM_EPOCHS)
    print("学习率：", LEARNING_RATE)

    # 如果checkpoints目录不存在，就自动创建。
    #
    # parents=True：
    # 如果上级目录缺失，也一并创建。
    #
    # exist_ok=True：
    # 如果目录已经存在，不要报错。
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 为本次实验生成时间名称。
    #
    # 例如：
    # baseline_20260916_153025
    #
    # 这样多次训练的TensorBoard日志不会混在一起。
    current_time = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # 每次实验生成独立的TensorBoard日志目录。
    #
    # 例如：
    # runs/label_smoothing_01_20260917_170000
    run_name = (
        f"{EXPERIMENT_NAME}_{current_time}"
    )
    log_dir = RUNS_DIR / run_name

    # 创建TensorBoard日志写入器。
    writer = SummaryWriter(
        log_dir=str(log_dir)
    )

    print("TensorBoard日志：", log_dir)
    print("最佳模型路径：", BEST_MODEL_PATH)

    # 创建DataLoader。
    #
    # 当前正式训练只使用训练集和验证集。
    # 测试集仍然保持封闭，等模型训练结束后再使用。
    train_loader, val_loader, _ = (
        create_dataloaders()
    )

    # 创建预训练ResNet18，并移动到GPU。
    model = build_model().to(device)

    # 创建交叉熵损失函数。
    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
    )

    print(
        "损失函数实际label_smoothing：",
        criterion.label_smoothing,
    )

    assert (
            criterion.label_smoothing
            == LABEL_SMOOTHING
    ), "损失函数没有正确应用标签平滑！"

    # 创建AdamW优化器。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # 记录目前最好的验证准确率。
    #
    # 设置成-1，确保第一个Epoch一定会保存。
    best_val_accuracy = -1.0

    try:
        # range(1, 11)会依次产生1～10。
        for epoch_number in range(
            1,
            NUM_EPOCHS + 1,
        ):
            print()
            print(
                f"========== "
                f"Epoch {epoch_number}/{NUM_EPOCHS} "
                f"=========="
            )

            # ================================================
            # 1. 训练一个Epoch
            # ================================================

            train_loss, train_accuracy = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch_number=epoch_number,
            )

            # ================================================
            # 2. 验证一个Epoch
            # ================================================

            val_loss, val_accuracy = validate_one_epoch(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                epoch_number=epoch_number,
            )

            # ================================================
            # 3. 打印当前Epoch结果
            # ================================================

            print()
            print(
                f"Epoch {epoch_number}结果："
            )
            print(
                f"Train Loss：{train_loss:.4f}"
            )
            print(
                f"Train Accuracy："
                f"{train_accuracy:.2%}"
            )
            print(
                f"Val Loss：{val_loss:.4f}"
            )
            print(
                f"Val Accuracy："
                f"{val_accuracy:.2%}"
            )

            # ================================================
            # 4. 写入TensorBoard
            # ================================================

            # 把训练损失和验证损失写在同一张图中。
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={
                    "Train": train_loss,
                    "Validation": val_loss,
                },
                global_step=epoch_number,
            )

            # 把训练准确率和验证准确率写在同一张图中。
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={
                    "Train": train_accuracy,
                    "Validation": val_accuracy,
                },
                global_step=epoch_number,
            )

            # 立即将本轮数据写入磁盘。
            writer.flush()

            # ================================================
            # 5. 判断是否保存最佳模型
            # ================================================

            # 只根据验证准确率选择模型。
            #
            # 不能根据测试集结果选择模型，
            # 否则会造成测试集信息泄漏。
            if val_accuracy > best_val_accuracy:
                old_best_accuracy = best_val_accuracy
                best_val_accuracy = val_accuracy

                # checkpoint不只保存模型参数，
                # 还保存当前Epoch、优化器状态和实验指标。
                checkpoint = {
                    "epoch": epoch_number,

                    # 模型学习到的参数。
                    "model_state_dict": (
                        model.state_dict()
                    ),

                    # 优化器内部状态。
                    # 以后如果需要续训，可以恢复。
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),

                    # 当前最佳验证结果。
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,

                    # 保存关键训练配置。
                    "learning_rate": LEARNING_RATE,
                    "num_classes": 37,
                    "model_name": "resnet18",
                    "experiment_name": EXPERIMENT_NAME,
                    "label_smoothing": LABEL_SMOOTHING,
                }

                torch.save(
                    checkpoint,
                    BEST_MODEL_PATH,
                )

                print(
                    "发现更好的模型："
                    f"{old_best_accuracy:.2%}"
                    " → "
                    f"{best_val_accuracy:.2%}"
                )
                print(
                    "已保存到：",
                    BEST_MODEL_PATH,
                )
            else:
                print(
                    "本轮未超过最佳验证准确率："
                    f"{best_val_accuracy:.2%}"
                )

    finally:
        # 无论训练正常结束还是中途出错，
        # 都尝试关闭TensorBoard写入器。
        writer.close()

    print()
    print("===== Baseline训练完成 =====")
    print(
        "最佳验证准确率：",
        f"{best_val_accuracy:.2%}",
    )
    print(
        "最佳模型保存位置：",
        BEST_MODEL_PATH,
    )
if __name__ == "__main__":
    main()