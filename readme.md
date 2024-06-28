### HashNet

PyTorch 实现 "HashNet: Deep Learning to Hash by Continuation"（ICCV 2017）

#### 运行环境

- Linux 或 OSX
- NVIDIA GPU + CUDA（可能需要 CuDNN）以及相应的 PyTorch 框架（版本 0.3.1）
- Python 2.7/3.5

#### 数据集

在实验中我们使用了flickr25、NUS-WIDE 和 COCO 数据集。coco数据集可以在 [这里](https://cocodataset.org/#download)下载，NUS-WIDE 点击[这里](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)下载，flickr25k数据集[这里](https://press.liacs.nl/mirflickr/mirdownload.html)在下载。相关的文件格式如下：

```c
	pytorch/
├── src/
│   ├── train.py
│   ├── test.py
│   ├── data_list.py
│   ├── network.py
│   └── 其他相关文件
├── data/
│   ├── coco/
│   │   ├── train.txt
│   │   ├── test.txt
│   │   ├── train2014/
│   │   │   └── (包含所有的图片文件)
│   │   └── val2014/
│   │       └── (包含所有的图片文件)
│   ├── flickr25k/
│   │   ├── train.txt
│   │   ├── test.txt
│   │   └── (相关数据和文件)
│   ├── NUSWIDE-10/
│       └── (相关数据和文件)
│   └── NUSWIDE-21/
│       └── (相关数据和文件)
├── snapshot
├── README.md        
```



首先，你可以手动下载 `torchvision` 库中介绍的 PyTorch 预训练模型，或者如果你已连接到互联网，可以自动下载。然后，你可以使用以下命令为每个数据集训练模型。

```
cd ../pytorch/src
python train.py --gpu_id 0 --dataset coco --prefix resnet50_hashnet --hash_bit 48 --net ResNet50 --lr 0.0003 --class_num 1.0
```

运行代码先cd到自己的文件目录，然后你可以设置命令参数来切换不同的实验。

- `gpu_id` 是运行实验的 GPU ID。可以根据自己的情况使用不同的GPU。
- `hash_bit` 参数是哈希码的位数，可以是32，48，64位等。
- `dataset` 是数据集选择。在我们的实验中，可以是 "flickr25k"、"NUSWIDE-10"、"NUSWIDE-21" 或 "coco"。
- `prefix` 是输出模型快照和日志文件的路径，在 "snapshot" 目录中。
- `net` 设置基础网络。有关设置的详细信息，可以查看 `network.py`。
  - 对于 AlexNet，"net" 是 AlexNet。
  - 对于 VGG 网络，"net" 类似于 VGG16。详细名称在 `network.py` 中。
  - 对于 ResNet，"net" 类似于 ResNet50。详细名称在 `network.py` 中。
- `lr` 是学习率。
- `class_num` 是正负样本对的平衡权重。

#### 评估

你可以使用以下命令评估每个数据集的平均精度均值（MAP）结果。

```
sh复制代码cd src
python test.py --gpu_id 0 --dataset coco --prefix resnet50_hashnet --hash_bit 48 --snapshot iter_09000
```

你可以设置命令参数来切换不同的实验。最终会输出模型在某数据集上的MAP值。

- `gpu_id` 是运行实验的 GPU ID。
- `hash_bit` 参数是哈希码的位数。
- `dataset` 是数据集选择。在我们的实验中，可以是"flickr25k"、"NUSWIDE-10"、"NUSWIDE-21" 或 "coco"。
- `iter_06000prefix` 是输出模型快照和日志文件的路径，会输出在 "snapshot" 目录中。
- `snapshot` 是快照模型名称。"iter_09000" 表示在迭代 9000 时保存的模型，同理还有"iter_06000"、"iter_03000"等。

