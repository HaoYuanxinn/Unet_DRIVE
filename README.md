# UNet Segmentation (PyTorch)

用 PyTorch 复现 UNet，使用DRIVE 数据集
主要参考仓库：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

## 项目结构
- `unet.py`：UNet 网络结构定义
- `train.py`：主训练与测试脚本
- `plot.py`：结果可视化
- `transforms.py`: 数据增强与预处理
- `datasets.py`:数据集定义
- `utils.py`:训练工具函数、指标与损失函数
- `DRIVE/`：DRIVE 数据集

## 环境依赖
- numpy==1.22.0
- pandas==1.4.4
- matplotlib==3.5.3
- Pillow

- Python3.10
- torch==1.13.1
- torchvision==0.14.1

