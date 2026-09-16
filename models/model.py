import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


# 本项目需要识别37个宠物品种。
NUM_CLASSES = 37


def build_model(num_classes=NUM_CLASSES):
    """
    创建用于宠物品种分类的ResNet18模型。

    参数：
        num_classes：
            模型最终需要输出多少个类别。
            默认值是37。

    返回：
        修改完成的ResNet18模型。
    """

    # DEFAULT表示使用torchvision推荐的最佳预训练权重。
    #
    # “预训练”表示ResNet18之前已经在ImageNet数据集上
    # 学习过大量通用图像特征，例如边缘、纹理、形状等。
    weights = ResNet18_Weights.DEFAULT

    # 创建ResNet18，并载入预训练参数。
    # 第一次运行时可能会自动下载约45MB的模型权重。
    model = resnet18(weights=weights)

    # ResNet18原来的最后一层用于ImageNet的1000分类。
    # model.fc表示最后一个全连接层。
    #
    # in_features表示这个全连接层接收多少个输入特征。
    # 对ResNet18而言通常是512。
    input_features = model.fc.in_features

    # 将原来的1000分类层替换成37分类层。
    #
    # nn.Linear(512, 37)表示：
    # 输入512个高级图像特征，
    # 输出37个类别分数。
    model.fc = nn.Linear(
        in_features=input_features,
        out_features=num_classes,
    )

    return model


def main():
    # 如果CUDA可用就使用GPU，否则使用CPU。
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 创建模型，并把模型参数移动到对应设备。
    model = build_model()
    model = model.to(device)

    # 当前只是检查模型，不进行训练。
    # eval模式会关闭训练阶段专用的行为。
    model.eval()

    # 创建4张假的测试图片。
    #
    # 形状含义：
    # 4：图片数量
    # 3：RGB通道
    # 224、224：图片尺寸
    #
    # 这里不是项目真实数据，
    # 只是检查模型能否接收正确形状的输入。
    dummy_images = torch.randn(
        4,
        3,
        224,
        224,
        device=device,
    )

    # no_grad表示当前不需要计算梯度。
    # 检查模型和正式评估时可以减少内存占用。
    with torch.no_grad():
        outputs = model(dummy_images)

    print("===== 模型检查 =====")
    print("运行设备：", device)
    print("模型最后一层：", model.fc)
    print("模拟输入形状：", dummy_images.shape)
    print("模型输出形状：", outputs.shape)

    # 4张图片，每张图片对应37个类别分数，
    # 因此输出必须是[4, 37]。
    assert outputs.shape == (4, NUM_CLASSES)

    print("模型输出形状正确！")


if __name__ == "__main__":
    main()