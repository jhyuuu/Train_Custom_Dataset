#!/usr/bin/env python
# coding: utf-8

# # 迁移学习微调训练图像分类模型
# 
# 在自己的图像分类数据集上，使用ImageNet预训练图像分类模型初始化，改动分类层，迁移学习微调训练
# 
# 同济子豪兄：https://space.bilibili.com/1900783
# 
# [代码运行云GPU环境](https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1)：GPU RTX 3060、CUDA v11.2

# ## 设置matplotlib中文字体

# In[16]:


# # windows操作系统
# plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签 
# plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号


# In[17]:


# Mac操作系统，参考 https://www.ngui.cc/51cto/show-727683.html
# 下载 simhei.ttf 字体文件
# !wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf


# In[1]:


# Linux操作系统，例如 云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 如果遇到 SSL 相关报错，重新运行本代码块即可
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf -O /environment/miniconda3/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf')
get_ipython().system('rm -rf /home/featurize/.cache/matplotlib')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font",family='SimHei') # 中文字体
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号


# ## 导入工具包

# In[2]:


import time
import os

import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")


# ## 获取计算硬件

# In[3]:


# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


# ## 图像预处理

# In[4]:


from torchvision import transforms

# 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ])


# ## 载入图像分类数据集

# In[5]:


# 数据集文件夹路径
dataset_dir = 'fruit30_split'


# In[6]:


train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'val')
print('训练集路径', train_path)
print('测试集路径', test_path)


# In[7]:


from torchvision import datasets

# 载入训练集
train_dataset = datasets.ImageFolder(train_path, train_transform)

# 载入测试集
test_dataset = datasets.ImageFolder(test_path, test_transform)


# In[11]:


print('训练集图像数量', len(train_dataset))
print('类别个数', len(train_dataset.classes))
print('各类别名称', train_dataset.classes)


# In[12]:


print('测试集图像数量', len(test_dataset))
print('类别个数', len(test_dataset.classes))
print('各类别名称', test_dataset.classes)


# ## 类别和索引号 一一对应

# In[13]:


# 各类别名称
class_names = train_dataset.classes
n_class = len(class_names)


# In[14]:


class_names


# In[15]:


# 映射关系：类别 到 索引号
train_dataset.class_to_idx


# In[16]:


# 映射关系：索引号 到 类别
idx_to_labels = {y:x for x,y in train_dataset.class_to_idx.items()}


# In[17]:


idx_to_labels


# In[18]:


# 保存为本地的 npy 文件
np.save('idx_to_labels.npy', idx_to_labels)
np.save('labels_to_idx.npy', train_dataset.class_to_idx)


# ## 定义数据加载器DataLoader

# In[19]:


from torch.utils.data import DataLoader


# In[20]:


BATCH_SIZE = 32

# 训练集的数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4
                         )

# 测试集的数据加载器
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=4
                        )


# ## 查看一个batch的图像和标注

# In[21]:


# DataLoader 是 python生成器，每次调用返回一个 batch 的数据
images, labels = next(iter(train_loader))


# In[22]:


images.shape


# In[23]:


labels


# ## 可视化一个batch的图像和标注

# In[24]:


# 将数据集中的Tensor张量转为numpy的array数据类型
images = images.numpy()


# In[25]:


images[5].shape


# In[26]:


plt.hist(images[5].flatten(), bins=50)
plt.show()


# In[27]:


# batch 中经过预处理的图像
idx = 2
plt.imshow(images[idx].transpose((1,2,0))) # 转为(224, 224, 3)
plt.title('label:'+str(labels[idx].item()))


# In[28]:


label = labels[idx].item()


# In[29]:


label


# In[30]:


pred_classname = idx_to_labels[label]


# In[31]:


pred_classname


# In[32]:


# 原始图像
idx = 2
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
plt.imshow(np.clip(images[idx].transpose((1,2,0)) * std + mean, 0, 1))
plt.title('label:'+ pred_classname)
plt.show()


# ## 导入训练需使用的工具包

# In[33]:


from torchvision import models
import torch.optim as optim


# ## 选择迁移学习训练方式
# 
# 斯坦福CS231N【迁移学习】中文精讲：https://www.bilibili.com/video/BV1K7411W7So
# 
# 斯坦福CS231N【迁移学习】官方笔记：https://cs231n.github.io/transfer-learning/

# ### 选择一：只微调训练模型最后一层（全连接分类层）

# In[34]:


model = models.resnet18(pretrained=True) # 载入预训练模型

# 修改全连接层，使得全连接层的输出与当前数据集类别数对应
# 新建的层默认 requires_grad=True
model.fc = nn.Linear(model.fc.in_features, n_class)


# In[35]:


model.fc


# In[36]:


# 只微调训练最后一层全连接层的参数，其它层冻结
optimizer = optim.Adam(model.fc.parameters())


# ### 选择二：微调训练所有层

# In[37]:


# model = models.resnet18(pretrained=True) # 载入预训练模型

# model.fc = nn.Linear(model.fc.in_features, n_class)

# optimizer = optim.Adam(model.parameters())


# ### 选择三：随机初始化模型全部权重，从头训练所有层

# In[38]:


# model = models.resnet18(pretrained=False) # 只载入模型结构，不载入预训练权重参数

# model.fc = nn.Linear(model.fc.in_features, n_class)

# optimizer = optim.Adam(model.parameters())


# ## 训练配置

# In[39]:


model = model.to(device)

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss() 

# 训练轮次 Epoch
EPOCHS = 20


# ## 模拟一个batch的训练

# In[40]:


# 获得一个 batch 的数据和标注
images, labels = next(iter(train_loader))
images = images.to(device)
labels = labels.to(device)


# In[43]:


# 输入模型，执行前向预测
outputs = model(images)


# In[44]:


# 获得当前 batch 所有图像的预测类别 logit 分数
outputs.shape


# In[45]:


# 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
loss = criterion(outputs, labels)


# In[47]:


# 反向传播“三部曲”
optimizer.zero_grad() # 清除梯度
loss.backward() # 反向传播
optimizer.step() # 优化更新


# In[48]:


# 获得当前 batch 所有图像的预测类别
_, preds = torch.max(outputs, 1)


# In[49]:


preds


# In[50]:


labels


# ## 运行完整训练

# In[51]:


# 遍历每个 EPOCH
for epoch in tqdm(range(EPOCHS)):

    model.train()

    for images, labels in train_loader:  # 获得一个 batch 的数据和标注
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels) # 计算当前 batch 中，每个样本的平均交叉熵损失函数值
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ## 在测试集上初步测试

# In[52]:


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum()

    print('测试集上的准确率为 {:.3f} %'.format(100 * correct / total))


# ## 保存模型

# In[53]:


torch.save(model, 'checkpoints/fruit30_pytorch_20220814.pth')


# ## 参考文档
# 
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# 
# https://www.bilibili.com/video/BV1qe4y1D7zD
# 
# https://www.bilibili.com/video/BV14J411X7Bb
# 
# https://www.bilibili.com/video/BV1w4411u7ay

# In[ ]:




