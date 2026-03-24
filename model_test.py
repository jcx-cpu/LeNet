import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet
import matplotlib.pyplot as plt



def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)
    return test_dataloader


def test_model_process(model, test_dataloader):
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # 讲模型放入到训练设备中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output= model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确度test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)

# ... existing code ...


def test_single_image(model, image_path):
    """
    测试单张图片

    Args:
        model: 加载好的模型
        image_path: 单张图片的路径
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.Grayscale(num_output_channels=1),  # 彩色转灰度
        transforms.ToTensor(),
    ])
    # FashionMNIST 标签映射
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 加载图片（支持彩色图）
    from PIL import Image
    image = Image.open(image_path)

    # 预处理并添加 batch 维度
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 前向传播
    with torch.no_grad():
        output = model(image_tensor)
        pre_lab = torch.argmax(output, dim=1).item()
        pre_label = class_names[pre_lab]

    print(f"图片路径：{image_path}")
    print(f"预测结果：{pre_label}")

    # 显示图片
    fig, ax = plt.subplots(figsize=(6, 6))

    display_image = Image.open(image_path)
    ax.imshow(display_image)

    ax.set_title("真实标签：待标注", fontsize=14, pad=10)
    ax.set_xlabel(f"预测：{pre_label}", fontsize=12, labelpad=10)

    # 隐藏坐标轴刻度和边框，但保留标签
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.show()

    return pre_lab,pre_label


if __name__=="__main__":
    # 加载模型
    model = LeNet()
    model.load_state_dict(torch.load('best_model.pth'))
    # 加载测试数据
    # test_dataloader = test_data_process()
    # # 加载模型测试的函数
    # #test_model_process(model, test_dataloader)
    # test_model_process(model, test_dataloader)
    # 方式 2: 测试单张图片
    test_single_image(model, 'T-shirt.png')

    """
    1 0.8739
    2 0.877
    3 0.829
    4 0.8391
    5 0.8476
    6 0.8619
    7 0.8594
    
    """