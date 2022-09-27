#!/usr/bin/env python
# coding: utf-8

# # 计算测试集图像语义特征
# 
# 抽取Pytorch训练得到的图像分类模型中间层的输出特征，作为输入图像的语义特征。
# 
# 计算测试集所有图像的语义特征，使用t-SNE和UMAP两种降维方法降维至二维和三维，可视化。
# 
# 分析不同类别的语义距离、异常数据、细粒度分类、高维数据结构。
# 
# 同济子豪兄：https://space.bilibili.com/1900783
# 
# [代码运行云GPU环境](https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1)：GPU RTX 3060、CUDA v11.2

# ## 导入工具包

# In[1]:


from tqdm import tqdm

import pandas as pd
import numpy as np

import torch

import cv2
from PIL import Image

# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


# ## 图像预处理

# In[2]:


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


# ## 导入训练好的模型

# In[3]:


model = torch.load('checkpoints/fruit30_pytorch_20220814.pth')
model = model.eval().to(device)


# ## 抽取模型中间层输出结果作为语义特征

# In[4]:


from torchvision.models.feature_extraction import create_feature_extractor


# In[6]:


model_trunc = create_feature_extractor(model, return_nodes={'avgpool': 'semantic_feature'})


# ## 计算单张图像的语义特征

# In[8]:


img_path = 'fruit30_split/val/菠萝/105.jpg'
img_pil = Image.open(img_path)
input_img = test_transform(img_pil) # 预处理
input_img = input_img.unsqueeze(0).to(device)
# 执行前向预测，得到指定中间层的输出
pred_logits = model_trunc(input_img) 


# In[9]:


pred_logits['semantic_feature'].squeeze().detach().cpu().numpy().shape


# In[19]:


# pred_logits['semantic_feature'].squeeze().detach().cpu().numpy()


# ## 载入测试集图像分类结果

# In[11]:


df = pd.read_csv('测试集预测结果.csv')


# In[12]:


df.head()


# ## 计算测试集每张图像的语义特征

# In[13]:


encoding_array = []
img_path_list = []

for img_path in tqdm(df['图像路径']):
    img_path_list.append(img_path)
    img_pil = Image.open(img_path).convert('RGB')
    input_img = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
    feature = model_trunc(input_img)['semantic_feature'].squeeze().detach().cpu().numpy() # 执行前向预测，得到 avgpool 层输出的语义特征
    encoding_array.append(feature)
encoding_array = np.array(encoding_array)


# In[14]:


encoding_array.shape


# ## 保存为本地的.npy文件

# In[20]:


# 保存为本地的 npy 文件
np.save('测试集语义特征.npy', encoding_array)


# In[ ]:




