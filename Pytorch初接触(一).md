谷歌云平台有Tensorflow和Pytorch的镜像，可以在创建磁盘的时候直接加载。

### 安装Pytorch
[安装链接](https://pytorch.org/get-started/locally/)

出现的错误：Torch和Torchvision的CUDA版本对不上

安装命令：
```
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
```
#### 疑问1：Torch和Torchvision的区别在哪儿？

Torchvision包是服务于Pytorch深度学习框架的，用来生成图片、视频数据集和一些流行的模型类和预训练模型，[参考链接](https://blog.csdn.net/tsq292978891/article/details/79403617)

#### 疑问2：Python中的@装饰器的作用？
[参考链接1](https://blog.csdn.net/star714/article/details/71045305)
[参考链接2](https://blog.csdn.net/u012759262/article/details/79749299)
[参考链接3](https://blog.51cto.com/zhuxianzhong/1604922)

ONNX支持：可以导出模型到ONNX(Open Neural Network Exchange)格式，然后用于部署之类的

Tensor和ndarray的区别在于，Tensor可以在GPU上实现加速运算

### 基础函数：
```
torch.empty()
torch.rand()
torch.zeros()
torch.tensor()
torch.randn_like()

torch.Size([5, 3])是一个tuple

torch.add(...., out=result)计算的返回结果赋值给result

a = torch.tensor([1])
b = torch.tensor([2])
b.add_(a)
b.copy_(a)
下划线_蛮有作用
Any operation that mutates a tensor in-place is post-fixed with an _

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
进行resize操作

If you have one element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
print(x.item())
```
100+ Tensor operations，包含transposing, indexing, slicing, mathematical operations, linear algebra, random numbers等等。[链接](https://pytorch.org/docs/stable/torch.html#)

Torch Tensor和Numpy array共用内存(如果Torch Tensor在CPU上)，不过CharTensor有点特殊？
```
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)将numpy转化成tensor
np.add(a, 1, out=a)
print(a, b)

把变量放到GPU上或者CPU上
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
```
### 自动微分
[简要介绍](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py)
[详细介绍](https://pytorch.org/docs/stable/autograd.html#)

torch.Tensor is the central class of the package. If you set its attribute `.requires_grad` as True, it starts to track all operations on it. When you finish your computation you can call `.backward()` and have all the gradients computed automatically. The gradient for this tensor will be accumulated into `.grad` attribute

To stop a tensor from tracking history, you can call `.detach` to detach it from the computation history, and to prevent future computation from being tracked

To prevent tracking history (and using memory), you can also wrap the code block in `with torch.no_grad()`. This can be particulary helpful when evaluating a model because the model may have trainable parameters with `requires_grad=True`, but for which we don't need the gradients

There's one more class which is very important for autograd implementation- a `Function`

If you want to compute the derivatives, you can call `.backward()` on a Tensor. If Tensor is a scalar (it holds a one element data), you don't need to specify any arguments to backward(), however if it has more elements, you need to specify a gradient argument that is a tensor of matching shape

```
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

其中的[0.1, 1.0, 0.0001]起到了缩放的作用？
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v, retain_graph=True)

print(x.grad)


print(x.requires_grad)
print((x ** 2).requires_grad)
包含在.no_grad()中的Tensor，gradient不起作用，在预测的时候很有用
with torch.no_grad():
    print((x ** 2).requires_grad)
```
### 搭建LeNet网络
训练流程：
+ Define the neural network that has some learnable parameters (or weights)
+ Iterate over a dataset of inputs
+ Process input through the network
+ Compute the loss (how far is the output from being correct)
+ Propagate gradients back into the network's parameters
+ Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient

[损失函数](https://pytorch.org/docs/stable/nn.html#loss-functions)
[super(Net, self).__init__()参考链接](https://blog.csdn.net/lqhbupt/article/details/19631991)

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 测试特征提取后的输出特征图大小
# input = torch.randn(1, 1, 32, 32)
#
# conv1 = nn.Conv2d(1, 6, 3)
# conv2 = nn.Conv2d(6, 16, 3)
#
# x = conv1(input)
# x = F.max_pool2d(x, 2)
# x = conv2(x)
# x = F.max_pool2d(x, 2)

# 定义网络, 类的继承
class Net(nn.Module):
    # 初始化
    def __init__(self):
        # 这个的作用得查一下
        super(Net, self).__init__()
        # 定义卷积操作
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 定义全连接操作
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
 # 网络的前向传播
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1: ]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
# 类的实例
net = Net()

input = torch.randn(1, 1, 32, 32)
# out = net(input)
# Zero the gradient buffers of all parameters and backprops with random gradients
# net.zero_grad()
# out.backward(torch.randn(1, 10))

# torch.nn只能支持mini-batches, 如果输入的维度为[1, 32, 32]可以使用input.unsqueeze_(0)变成[1, 1, 32, 32]
target = torch.randn(1, 10)
criterion = nn.MSELoss()
# loss = criterion(out, target)
#
# net.zero_grad()
# print(net.conv1.bias.grad)
# loss.backward(retain_graph=True)
# print(net.conv1.bias.grad)

# 网络的可学习参数由net.parameters()返回
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

### 数据加载
先把数据加载到numpy，再转化成torch.Tensor：
+ For images, packages such as Pillow, OpenCV are useful
+ For audio, packages such as scipy and librosa
+ For text, either raw Python or Cython based loading, or NLTK and SpaCy are useful

### 训练一个分类器
步骤：
+ Load and normalizing the CIFAR10 training and test datasets using torchvision
+ Define a Convolutional Neural Network
+ Define a loss function
+ Train the network on the training data
+ Test the network on the test data

```
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
#
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # print("In Model: input batch_size", x.size())
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # print("In Model: output batch_size", x.size())
        return x


net = Net()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(5): # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)其中outputs=[4, 10]，labels=[4]
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999: # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            print("Outside: input batch_size", inputs.size(),
                  "output batch_size", outputs.size())

print('Finished Training')

# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
#
# correct = 0
# total = 0
# with torch.no_grad():
# for data in testloader:
# images, labels = data
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# total += labels.size()[0]
# correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
#
class_correct = list(0.0 for i in range(10))
class_total = list(0.0 for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
```

构建一个数据加载器：
```
size1, size2, size3 = (3, 32, 32)
# output_size = (10)

batch_size = 4
data_size = 50000

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomDataset(Dataset):

    def __init__(self, size1, size2, size3, length):
        self.len = length
        self.data = torch.rand(length, size1, size2, size3, dtype=torch.float)
        self.label = torch.randint(0, 9, size=(length, 1))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(size1, size2, size3, data_size),
                         batch_size=batch_size, shuffle=True, num_workers=1)
```
