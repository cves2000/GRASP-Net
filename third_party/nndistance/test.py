import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from torch.autograd import Variable  # 从 torch.autograd 导入 Variable 类

from modules.nnd import NNDModule  # 从 modules.nnd 导入 NNDModule 类

dist =  NNDModule()  # 创建一个 NNDModule 的实例

p1 = torch.rand(10,1000,3)  # 创建一个形状为 (10, 1000, 3) 的随机张量
p2 = torch.rand(10,1500,3)  # 创建一个形状为 (10, 1500, 3) 的随机张量
points1 = Variable(p1,requires_grad = True)  # 创建一个可求导的 Variable
points2 = Variable(p2)  # 创建一个 Variable
points1=points1.cuda()  # 将 points1 移到 GPU
points2=points2.cuda()  # 将 points2 移到 GPU
dist1, dist2, idx1, idx2 = dist(points1, points2)  # 计算 points1 和 points2 之间的距离和索引
print(dist1, dist2, idx1, idx2)  # 打印距离和索引
loss = torch.sum(dist1)  # 计算 loss，这里简单地将 dist1 的所有元素求和
print(loss)  # 打印 loss
loss.backward()  # 对 loss 进行反向传播
print(points1.grad, points2.grad)  # 打印 points1 和 points2 的梯度


points1 = Variable(p1.cuda(), requires_grad = True)  # 创建一个可求导的 Variable，并将其移动到 GPU
points2 = Variable(p2.cuda())  # 创建一个 Variable，并将其移动到 GPU
dist1, dist2, idx1, idx2 = dist(points1, points2)  # 计算 points1 和 points2 之间的距离和索引
print(dist1, dist2, idx1, idx2)  # 打印距离和索引
loss = torch.sum(dist1)  # 计算 loss，这里简单地将 dist1 的所有元素求和
print(loss)  # 打印 loss
loss.backward()  # 对 loss 进行反向传播
print(points1.grad, points2.grad)  # 打印 points1 和 points2 的梯度

# 测试索引
nn2 = torch.gather(points1, 1, idx2.unsqueeze(-1).expand([-1,-1,points1.shape[2]]).cuda())  # 使用 torch.gather 函数根据 idx2 对 points1 进行索引
print(nn2)  # 打印 nn2
loss = torch.sum(nn2)  # 计算 loss，这里简单地将 nn2 的所有元素求和
print(loss)  # 打印 loss
loss.backward()  # 对 loss 进行反向传播
print(points1.grad, points2.grad)  # 打印 points1 和 points2 的梯度

# 这段代码的主要功能是计算两组点云之间的最近邻距离（Nearest Neighbor Distance，NND）。首先，它创建了两组随机点云，然后使用 NNDModule 计算了这两组点云之间的最近邻距离和对应的索引。
# 然后，它计算了一个损失函数，即所有最近邻距离的和，并对这个损失函数进行了反向传播，以计算点云的梯度。这个过程被执行了两次。最后，它测试了使用索引来收集点云的功能。
# 这段代码可能用于点云处理的任务中，例如点云的配准、分类或分割等。
