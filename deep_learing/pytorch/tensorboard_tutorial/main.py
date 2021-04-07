'''
1.Read in data and with appropriate transforms (nearly identical to the prior tutorial).
2.Set up TensorBoard.
3.Write to TensorBoard.
4.Inspect a model architecture using TensorBoard.
5.Use TensorBoard to create interactive versions of the visualizations we created in last tutorial, with less code
'''
# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# transforms 数据转换
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)

# 数据集 datasets
trainset = torchvision.datasets.FashionMNIST('./data',
                                             download=True,
                                             train=True,
                                             transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)


# dataloaders 数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=0)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=0)

# constant for classes 类别的内容
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
# 辅助展示一张图片
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim = 0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.show(np.transpose(npimg,(1,2,0)))

# 图像是单通道 28*28
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6 , 5) #(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # pooling层没有参数
        x = self.pool(F.relu(self.conv2(x)))
        # 数据拉直
        x = x.view(-1,16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# 定义优化器和准则(损失函数) optimizer , criterion
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

'''
======================================TensorBoard==========================
tensorboard --logdir=runs
'''
# 1 TensorBoard setup设置

from torch.utils.tensorboard import SummaryWriter
# 定义日志路径
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
# 2 写入TensorBoard

# 随机选取训练图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 创建图片网格
img_grid = torchvision.utils.make_grid(images)

# 图片展示
matplotlib_imshow(img_grid, one_channel=True)

#写入tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)

# 运行一下命令 tensorboard --logdir=runs

# 3 使用TensorBoard检查模型
writer.add_graph(net, images)
writer.close()

'''
4. 为TensorBoard添加一个投影仪Projector
我们可以通过add_embedding方法可视化高维数据的低维表示
'''
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
# 辅助方法
def select_n_random(data, labels, n=100):
    '''
    随机选择n个数据和他们所对应的label从一个数据集中
    '''
    assert len(data) == len(labels)

    # randperm 随机排列组合
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# 随机选择图片和她对应的索引
images, labels = select_n_random(trainset.data, trainset.targets)

# 得到每一个图片的label
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28*28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.close()

'''
问题记录
AttributeError: module 'tensorflow._api.v2.io.gfile' has no attribute 'get_filesystem'
解决办法就是：在同一个环境中只安装PyTorch和TensorBoard即可。
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
'''

'''
5. 模型训练追踪用tensorboard
'''

# def images_to_probs(net, images):
#     '''
#
#     '''
#     output = net(images)
#     _, preds_tensor = torch.max(output, 1) # 返回2维向量每个维度里的最大值
#     # convert output probabilities to predicted class将输出的概率转换成预测的标签
#     preds = np.squeeze(preds_tensor.numpy())
#     return preds, [F.softmax(el, dim=0).item() for i, el in zip(preds, output)]
#


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

# def plot_class_preds(net, images, labels):
#
#     preds, probs = images_to_probs(net, images)
#     # 画出在每个batch里的图片，预测值和标签
#     fig = plt.figure(figsize=(12, 48))
#     for idx in np.arange(4):
#         ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
#         matplotlib_imshow(images[idx], one_channel=True)
#         ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
#             classes[preds[idx]],
#             probs[idx] * 100,
#             classes[labels[idx]]),
#             color=("green" if preds[idx]==labels[idx].item() else "red")
#         )
#     return fig
def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig
running_loss = 0.0
for epoch in range(1):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        #梯度清零
        optimizer.zero_grad()

        #前向传播
        outputs = net(inputs)
        #loss
        loss  = criterion(outputs, labels)
        #反向传播
        loss.backward()
        #梯度更新
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999: # 每1000个mini-batch

            # 记录running loss
            writer.add_scalar('training loss',
                              running_loss / 1000,
                              epoch * len(trainloader)+ i)
            # 记录
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(net, inputs, labels),
                              global_step=epoch * len(trainloader) + i)
            running_loss = 0.0

print('Finished Training')

'''
6. 评估训练模型用TensorBoard
1 在test_size x 中获得预测的概率
2 获得test_size 的 预测标签
'''
class_probs = []
class_preds = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)

        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)

# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    '''
    class_index 从0-9
    绘制相应的 precision-recall curve
    '''

    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_preds)