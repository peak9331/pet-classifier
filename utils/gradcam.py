"""生成一张预测正确和一张预测错误的 Bengal Grad-CAM 可视化。"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import create_dataloaders
from utils.checkpoints import CLASS_NAMES, load_trained_model


PROJECT_ROOT = Path(__file__).resolve().parent[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "label_smoothing_01_best.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs"

BENGAL_INDEX = CLASS_NAMES.index("Bengal")
EGYPTIAN_MAU_INDEX = CLASS_NAMES.index("Egyptian_Mau")


class GradCAM:
    """为 ResNet18 最后一组卷积层生成 Grad-CAM 热力图。"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        module_input,
        module_output,
    ):
        """保存目标卷积层输出的特征图。"""

        self.activations = module_output.detach()

        if module_output.requires_grad:
            module_output.register_hook(
                self._save_gradient
            )

    def _save_gradient(self, gradient):
        """保存目标类别对特征图的梯度。"""

        self.gradients = gradient.detach()

    @torch.enable_grad()
    def generate(
        self,
        input_tensor,
        target_class,
    ):
        """
        为指定类别生成热力图。

        参数：
            input_tensor：
                一张形状为 [1, 3, 224, 224] 的图片。

            target_class：
                需要解释的类别编号。

        返回：
            heatmap：
                数值范围为 0～1 的二维热力图。

            logits：
                模型输出的 37 个类别分数。
        """

        if (
            input_tensor.ndim != 4
            or input_tensor.size(0) != 1
        ):
            raise ValueError(
                "Grad-CAM 每次必须接收一张 "
                "[1, C, H, W] 图片。"
            )

        self.model.eval()
        self.activations = None
        self.gradients = None

        input_tensor = (
            input_tensor
            .detach()
            .requires_grad_(True)
        )

        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        if not 0 <= target_class < logits.size(1):
            raise ValueError(
                f"目标类别编号超出范围：{target_class}"
            )

        target_score = logits[0, target_class]
        target_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "没有获取到目标层的特征图。"
            )

        if self.gradients is None:
            raise RuntimeError(
                "没有获取到目标层的梯度。"
            )

        # 在特征图的高、宽方向求平均，
        # 得到每个通道的重要程度。
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # 使用通道权重对特征图加权求和。
        cam = (
            weights * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        # 只保留对目标类别有正向贡献的部分。
        cam = torch.relu(cam)

        # 放大到输入图片尺寸。
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        # 归一化到 0～1。
        cam = cam - cam.min()

        maximum = cam.max()

        if maximum > 0:
            cam = cam / maximum

        return (
            cam.cpu().numpy(),
            logits.detach(),
        )

    def remove_hooks(self):
        """分析结束后移除前向钩子。"""

        self.forward_handle.remove()


def denormalize_image(image_tensor):
    """撤销 ImageNet 标准化，恢复适合显示的图片。"""

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = (
        image_tensor.detach().cpu() * std
        + mean
    )

    image = image.clamp(0, 1)

    return image.permute(1, 2, 0).numpy()


def save_gradcam_figure(
    image_tensor,
    heatmap,
    true_class,
    predicted_class,
    output_path,
    figure_title,
    overwrite=False,
):
    """保存原图、热力图和叠加图。"""

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        print(
            f"图片已存在，保持不变：{output_path}"
        )
        return output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_image = denormalize_image(
        image_tensor
    )

    colored_heatmap = plt.get_cmap(
        "jet"
    )(heatmap)[..., :3]

    overlay = (
        0.60 * original_image
        + 0.40 * colored_heatmap
    )

    overlay = np.clip(overlay, 0, 1)

    true_name = CLASS_NAMES[true_class]
    predicted_name = CLASS_NAMES[predicted_class]

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

    figure.tight_layout()

    try:
        mode = "wb" if overwrite else "xb"

        with output_path.open(mode) as stream:
            figure.savefig(
                stream,
                format="png",
                dpi=300,
                bbox_inches="tight",
            )
    finally:
        plt.close(figure)

    print("图片已保存：", output_path)

    return output_path


def find_bengal_samples(
    model,
    test_loader,
    device,
):
    """
    从测试集中寻找：

    1. 一张正确预测的 Bengal；
    2. 一张被预测为 Egyptian Mau 的 Bengal。
    """

    model.eval()

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

                if (
                    correct_sample is None
                    and true_class == BENGAL_INDEX
                    and predicted_class == BENGAL_INDEX
                ):
                    correct_sample = (
                        images[index].clone(),
                        true_class,
                        predicted_class,
                    )

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
        "没有找到所需的 Bengal 样本。"
        "当前模型可能没有把 Bengal 误判为 Egyptian Mau。"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "为 Label Smoothing 模型生成 "
            "Bengal Grad-CAM 可视化"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "模型 checkpoint 路径，默认读取 "
            "checkpoints/label_smoothing_01_best.pth"
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader 工作进程数量",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有 Grad-CAM 图片",
    )

    args = parser.parse_args()

    if args.num_workers < 0:
        parser.error(
            "--num-workers 不能小于 0"
        )

    return args


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("===== Grad-CAM 分析 =====")
    print("运行设备：", device)
    print("模型路径：", args.checkpoint)

    model, checkpoint = load_trained_model(
        args.checkpoint,
        device,
    )

    print(
        "模型最佳 Epoch：",
        checkpoint.get("epoch"),
    )

    print(
        "模型验证准确率：",
        f"{checkpoint.get('val_accuracy', 0):.2%}",
    )

    _, _, test_loader = create_dataloaders(
        download=False,
        num_workers=args.num_workers,
    )

    correct_sample, incorrect_sample = (
        find_bengal_samples(
            model,
            test_loader,
            device,
        )
    )

    # ResNet18 的 layer4[-1] 是最后一个残差块，
    # 适合用于生成 Grad-CAM。
    gradcam = GradCAM(
        model,
        model.layer4[-1],
    )

    cases = [
        (
            correct_sample,
            "gradcam_correct_bengal.png",
            "Correct prediction",
        ),
        (
            incorrect_sample,
            (
                "gradcam_incorrect_bengal_"
                "as_egyptian_mau.png"
            ),
            "Incorrect prediction",
        ),
    ]

    try:
        for sample, filename, title in cases:
            image, true_class, predicted_class = sample

            heatmap, _ = gradcam.generate(
                image.unsqueeze(0).to(device),
                predicted_class,
            )

            if not np.isfinite(heatmap).all():
                raise ValueError(
                    "Grad-CAM 热力图中出现了非有限值。"
                )

            save_gradcam_figure(
                image_tensor=image,
                heatmap=heatmap,
                true_class=true_class,
                predicted_class=predicted_class,
                output_path=OUTPUT_DIR / filename,
                figure_title=title,
                overwrite=args.overwrite,
            )

            print(
                "Grad-CAM 验证完成：",
                f"{CLASS_NAMES[true_class]} → "
                f"{CLASS_NAMES[predicted_class]}",
            )
    finally:
        gradcam.remove_hooks()

    print("===== Grad-CAM 分析完成 =====")


if __name__ == "__main__":
    main()
