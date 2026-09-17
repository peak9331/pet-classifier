from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import create_dataloaders
from evaluate import CLASS_NAMES, load_trained_model


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# 使用不会被train.py覆盖的完整训练备份。
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "label_smoothing_01_completed.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Bengal的类别编号为5。
BENGAL_INDEX = CLASS_NAMES.index("Bengal")

# Egyptian Mau的类别编号为11。
EGYPTIAN_MAU_INDEX = CLASS_NAMES.index(
    "Egyptian_Mau"
)


# ============================================================
# 2. Grad-CAM计算器
# ============================================================

class GradCAM:
    """
    从ResNet18最后一组卷积层提取Grad-CAM热力图。
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # 注册前向传播钩子，用于保存特征图。
        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        module_input,
        module_output,
    ):
        # 保存卷积层输出的特征图。
        self.activations = module_output.detach()

        # 当反向传播经过该特征图时，保存梯度。
        module_output.register_hook(
            self._save_gradient
        )

    def _save_gradient(self, gradient):
        self.gradients = gradient.detach()

    def generate(
        self,
        input_tensor,
        target_class,
    ):
        """
        为指定类别生成Grad-CAM。

        input_tensor形状：
            [1, 3, 224, 224]

        target_class：
            需要解释的类别编号。
        """

        # 清除上一次反向传播留下的梯度。
        self.model.zero_grad(
            set_to_none=True
        )

        # 前向传播，得到37个类别分数。
        logits = self.model(input_tensor)

        # 取出目标类别的分数。
        target_score = logits[
            0,
            target_class,
        ]

        # 反向传播，计算目标类别对卷积特征的梯度。
        target_score.backward()

        if self.activations is None:
            raise RuntimeError("没有获取到特征图。")

        if self.gradients is None:
            raise RuntimeError("没有获取到梯度。")

        # 在特征图的高和宽方向求平均，
        # 得到每个通道的重要程度。
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # 对所有通道加权求和。
        cam = (
            weights * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        # 只保留对目标类别有正向贡献的区域。
        cam = torch.relu(cam)

        # 将小尺寸热力图放大到224×224。
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # 移除Batch和Channel维度。
        cam = cam[0, 0]

        # 将数值缩放到0～1之间。
        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        return (
            cam.cpu().numpy(),
            logits.detach(),
        )

    def remove_hooks(self):
        # 分析结束后移除钩子。
        self.forward_handle.remove()


# ============================================================
# 3. 寻找需要分析的两张图片
# ============================================================

def find_bengal_samples(
    model,
    test_loader,
    device,
):
    """
    从测试集中寻找：

    1. 一张正确预测为Bengal的图片；
    2. 一张被错误预测为Egyptian Mau的Bengal图片。
    """

    correct_sample = None
    incorrect_sample = None

    with torch.no_grad():
        for images, labels in test_loader:
            images_on_device = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images_on_device)

            predictions = logits.argmax(dim=1)

            for index in range(len(labels)):
                true_class = labels[index].item()
                predicted_class = (
                    predictions[index].item()
                )

                # 寻找正确识别的Bengal。
                if (
                    correct_sample is None
                    and true_class == BENGAL_INDEX
                    and predicted_class
                    == BENGAL_INDEX
                ):
                    correct_sample = (
                        images[index].clone(),
                        true_class,
                        predicted_class,
                    )

                # 寻找被误认为Egyptian Mau的Bengal。
                if (
                    incorrect_sample is None
                    and true_class == BENGAL_INDEX
                    and predicted_class
                    == EGYPTIAN_MAU_INDEX
                ):
                    incorrect_sample = (
                        images[index].clone(),
                        true_class,
                        predicted_class,
                    )

                if (
                    correct_sample is not None
                    and incorrect_sample is not None
                ):
                    return (
                        correct_sample,
                        incorrect_sample,
                    )

    raise RuntimeError(
        "没有在测试集中找到需要的Bengal样本。"
    )


# ============================================================
# 4. 将标准化图片恢复为可显示图片
# ============================================================

def denormalize_image(image_tensor):
    """
    撤销ImageNet Normalize，恢复图片颜色。
    """

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = (
        image_tensor.cpu() * std + mean
    )

    image = image.clamp(0, 1)

    # 从[C, H, W]转换成[H, W, C]。
    image = image.permute(
        1,
        2,
        0,
    ).numpy()

    return image


# ============================================================
# 5. 保存Grad-CAM图片
# ============================================================

def save_gradcam_figure(
    image_tensor,
    heatmap,
    true_class,
    predicted_class,
    output_path,
    figure_title,
):
    """
    保存原图、热力图和叠加结果。
    """

    original_image = denormalize_image(
        image_tensor
    )

    # 把0～1的热力图转换成彩色热力图。
    colored_heatmap = plt.get_cmap(
        "jet"
    )(heatmap)[..., :3]

    # 将原图和彩色热力图叠加。
    overlay = (
        0.60 * original_image
        + 0.40 * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1,
    )

    true_name = CLASS_NAMES[true_class]
    predicted_name = CLASS_NAMES[
        predicted_class
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    figure.suptitle(
        (
            f"{figure_title}\n"
            f"True: {true_name} | "
            f"Predicted: {predicted_name}"
        ),
        fontsize=14,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# 6. 主程序
# ============================================================

def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("===== Grad-CAM分析 =====")
    print("运行设备：", device)
    print("模型路径：", CHECKPOINT_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 只使用测试集，不进行训练。
    _, _, test_loader = create_dataloaders()

    model, checkpoint = load_trained_model(
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    print(
        "模型最佳Epoch：",
        checkpoint.get("epoch"),
    )
    print(
        "模型验证准确率：",
        f"{checkpoint.get('val_accuracy'):.2%}",
    )

    # 先寻找正确和错误样本。
    correct_sample, incorrect_sample = (
        find_bengal_samples(
            model=model,
            test_loader=test_loader,
            device=device,
        )
    )

    # ResNet18最后一个卷积残差块。
    target_layer = model.layer4[-1]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    # --------------------------------------------------------
    # 正确样本：Bengal → Bengal
    # --------------------------------------------------------

    (
        correct_image,
        correct_true_class,
        correct_predicted_class,
    ) = correct_sample

    correct_input = (
        correct_image.unsqueeze(0).to(device)
    )

    correct_heatmap, _ = gradcam.generate(
        input_tensor=correct_input,
        target_class=correct_predicted_class,
    )

    correct_output_path = (
        OUTPUT_DIR
        / "gradcam_correct_bengal.png"
    )

    save_gradcam_figure(
        image_tensor=correct_image,
        heatmap=correct_heatmap,
        true_class=correct_true_class,
        predicted_class=(
            correct_predicted_class
        ),
        output_path=correct_output_path,
        figure_title="Correct prediction",
    )

    # --------------------------------------------------------
    # 错误样本：Bengal → Egyptian Mau
    # --------------------------------------------------------

    (
        incorrect_image,
        incorrect_true_class,
        incorrect_predicted_class,
    ) = incorrect_sample

    incorrect_input = (
        incorrect_image.unsqueeze(0).to(device)
    )

    # 对错误预测出来的Egyptian Mau类别进行解释。
    incorrect_heatmap, _ = gradcam.generate(
        input_tensor=incorrect_input,
        target_class=(
            incorrect_predicted_class
        ),
    )

    incorrect_output_path = (
        OUTPUT_DIR
        / (
            "gradcam_incorrect_"
            "bengal_as_egyptian_mau.png"
        )
    )

    save_gradcam_figure(
        image_tensor=incorrect_image,
        heatmap=incorrect_heatmap,
        true_class=incorrect_true_class,
        predicted_class=(
            incorrect_predicted_class
        ),
        output_path=incorrect_output_path,
        figure_title="Incorrect prediction",
    )

    gradcam.remove_hooks()

    print()
    print("正确样本：Bengal → Bengal")
    print("错误样本：Bengal → Egyptian Mau")
    print(
        "正确样本Grad-CAM已保存：",
        correct_output_path,
    )
    print(
        "错误样本Grad-CAM已保存：",
        incorrect_output_path,
    )
    print("===== Grad-CAM分析完成 =====")


if __name__ == "__main__":
    main()
