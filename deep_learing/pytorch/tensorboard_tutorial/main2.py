import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


from torch.utils.tensorboard import SummaryWriter
# 定义日志路径
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
# 2 写入TensorBoard

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
dataset = torchvision.datasets.MNIST(root='../../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=100,
                                          shuffle=True)

class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forword(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)
total_step = 50000

# start training
for step in range(total_step):

    # reset the data_iter
    if (step+1) % iter_per_epoch == 0:
        data_iter = iter(data_loader)

    # fetch images ands labels 绑定特征和标签,GPU
    images, labels = next(data_loader)
    images, labels = images.view(images.size(0), -1).to(device), labels.to(device)

    #forward pass前向传播
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward and optimize 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #compute accuracy
    _, argmax =  torch.max(outputs, 1 )
    accuracy = (labels == argmax.squeeze()).float().mean()
    # accuracy = (labels == argmax.squeeze()).float().mean()

    if(step+1) % 100 == 0 :
        print('Step[{}/{}], Loss: {:.4f}, Acc: {:.2f}'
              .format(step+1, total_step, loss.item(), accuracy.item()))
    '''
    ==================================TensorBoard===================================
    '''
    # 2. Log values and gradients of th parameters(histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.','/')
        writer.add_histogram(tag, value.data.cpu().numpy(), step+1)
        writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), step+1,)
        # tag value.data.cpu().numpy(),step+1
        # tag+'/grad' value.grad.data.cpu().numpy(), step+1
