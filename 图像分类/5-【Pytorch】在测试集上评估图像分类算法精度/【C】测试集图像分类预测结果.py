#!/usr/bin/env python
# coding: utf-8

# # 测试集图像分类预测结果
# 
# 使用训练好的图像分类模型，预测测试集的所有图像，得到预测结果表格。
# 
# 同济子豪兄：https://space.bilibili.com/1900783
# 
# [代码运行云GPU环境](https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1)：GPU RTX 3060、CUDA v11.2

# ## 导入工具包

# In[2]:


import os
from tqdm import tqdm

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn.functional as F

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


# ## 图像预处理

# In[4]:


from torchvision import transforms

# # 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
# train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
#                                       transforms.RandomHorizontalFlip(),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                      ])

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ])


# ## 载入测试集（和训练代码教程相同）

# In[6]:


# 数据集文件夹路径
dataset_dir = 'fruit30_split'
test_path = os.path.join(dataset_dir, 'val')
from torchvision import datasets
# 载入测试集
test_dataset = datasets.ImageFolder(test_path, test_transform)
print('测试集图像数量', len(test_dataset))
print('类别个数', len(test_dataset.classes))
print('各类别名称', test_dataset.classes)
# 载入类别名称 和 ID索引号 的映射字典
idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
# 获得类别名称
classes = list(idx_to_labels.values())
print(classes)


# ## 导入训练好的模型

# In[7]:


model = torch.load('checkpoints/fruit30_pytorch_20220814.pth')
model = model.eval().to(device)


# ## 表格A-测试集图像路径及标注

# In[8]:


test_dataset.imgs[:10]


# In[9]:


img_paths = [each[0] for each in test_dataset.imgs]


# In[10]:


df = pd.DataFrame()
df['图像路径'] = img_paths
df['标注类别ID'] = test_dataset.targets
df['标注类别名称'] = [idx_to_labels[ID] for ID in test_dataset.targets]


# In[11]:


df


# ## 表格B-测试集每张图像的图像分类预测结果，以及各类别置信度

# In[12]:


# 记录 top-n 预测结果
n = 3


# In[13]:


df_pred = pd.DataFrame()
for idx, row in tqdm(df.iterrows()):
    img_path = row['图像路径']
    img_pil = Image.open(img_path).convert('RGB')
    input_img = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
    pred_logits = model(input_img) # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算

    pred_dict = {}

    top_n = torch.topk(pred_softmax, n) # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze() # 解析出类别
    
    # top-n 预测结果
    for i in range(1, n+1):
        pred_dict['top-{}-预测ID'.format(i)] = pred_ids[i-1]
        pred_dict['top-{}-预测名称'.format(i)] = idx_to_labels[pred_ids[i-1]]
    pred_dict['top-n预测正确'] = row['标注类别ID'] in pred_ids
    # 每个类别的预测置信度
    for idx, each in enumerate(classes):
        pred_dict['{}-预测置信度'.format(each)] = pred_softmax[0][idx].cpu().detach().numpy()
        
    df_pred = df_pred.append(pred_dict, ignore_index=True)


# In[14]:


df_pred


# ## 拼接AB两张表格

# In[15]:


df = pd.concat([df, df_pred], axis=1)


# In[16]:


df


# ## 导出完整表格

# In[17]:


df.to_csv('测试集预测结果.csv', index=False)


# In[ ]:




