#!/usr/bin/env python
# coding: utf-8

# # ImageNet预训练图像分类模型预测单张图像-中文
# 
# 使用 ImageNet 预训练图像分类模型，对单张图像文件执行前向预测。
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
matplotlib.rc("font",family='SimHei') # 中文字体


# ## 导入pillow中文字体

# In[2]:


from PIL import ImageFont, ImageDraw
# 导入中文字体，指定字号
font = ImageFont.truetype('SimHei.ttf', 32)


# ## 导入工具包

# In[3]:


import os

import cv2
from PIL import Image, ImageFont, ImageDraw

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


# ## 预训练图像分类模型

# In[4]:


# 载入预训练图像分类模型

model = models.resnet18(pretrained=True) 

# model = models.resnet152(pretrained=True)

model = model.eval()
model = model.to(device)


# ## 载入ImageNet 1000图像分类中文标签

# In[5]:


df = pd.read_csv('imagenet_class_index.csv')
idx_to_labels = {}
for idx, row in df.iterrows():
    idx_to_labels[row['ID']] = [row['wordnet'], row['Chinese']]


# In[7]:


# idx_to_labels


# ## 图像预处理

# In[8]:


# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ])


# ## 载入一张测试图像

# In[9]:


# img_path = 'test_img/banana1.jpg'
# img_path = 'test_img/husky1.jpeg'
# img_path = 'test_img/watermelon1.jpg'
img_path = 'test_img/cat_dog.jpg'

img_pil = Image.open(img_path) # 用 pillow 载入


# In[10]:


img_pil


# ## 执行图像分类预测

# In[11]:


input_img = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
pred_logits = model(input_img) # 执行前向预测，得到所有类别的 logit 预测分数
pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算


# ## 各类别置信度柱状图

# In[12]:


plt.figure(figsize=(8,4))

x = range(1000)
y = pred_softmax.cpu().detach().numpy()[0]

ax = plt.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
plt.ylim([0, 1.0]) # y轴取值范围
# plt.bar_label(ax, fmt='%.2f', fontsize=15) # 置信度数值

plt.title(img_path, fontsize=30)
plt.xlabel('类别', fontsize=20)
plt.ylabel('置信度', fontsize=20)
plt.tick_params(labelsize=16) # 坐标文字大小

plt.show()


# ## 取置信度最大的 n 个结果

# In[13]:


n = 10
top_n = torch.topk(pred_softmax, n) # 取置信度最大的 n 个结果
pred_ids = top_n[1].cpu().detach().numpy().squeeze() # 解析出类别
confs = top_n[0].cpu().detach().numpy().squeeze() # 解析出置信度


# ## 图像分类结果写在原图上

# In[14]:


draw = ImageDraw.Draw(img_pil)


# In[15]:


for i in range(n):
    class_name = idx_to_labels[pred_ids[i]][1] # 获取类别名称
    confidence = confs[i] * 100 # 获取置信度
    text = '{:<15} {:>.4f}'.format(class_name, confidence)
    print(text)
    
    # 文字坐标，中文字符串，字体，rgba颜色
    draw.text((50, 100 + 50 * i), text, font=font, fill=(255, 0, 0, 1))


# In[16]:


img_pil


# In[17]:


# 保存图像
img_pil.save('output/img_pred.jpg')


# ### 图像和柱状图一起显示

# In[18]:


fig = plt.figure(figsize=(18,6))

# 绘制左图-预测图
ax1 = plt.subplot(1,2,1)
ax1.imshow(img_pil)
ax1.axis('off')

# 绘制右图-柱状图
ax2 = plt.subplot(1,2,2)
x = df['ID']
y = pred_softmax.cpu().detach().numpy()[0]
ax2.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
plt.ylim([0, 1.0]) # y轴取值范围
plt.xlabel('类别', fontsize=20)
plt.ylabel('置信度', fontsize=20)
ax2.tick_params(labelsize=16) # 坐标文字大小

plt.title('{} 图像分类预测结果'.format(img_path), fontsize=30)

plt.tight_layout()
fig.savefig('output/预测图+柱状图.jpg')


# ### 预测结果表格输出

# In[19]:


pred_df = pd.DataFrame() # 预测结果表格
for i in range(n):
    class_name = idx_to_labels[pred_ids[i]][1] # 获取类别名称
    label_idx = int(pred_ids[i]) # 获取类别号
    wordnet = idx_to_labels[pred_ids[i]][0] # 获取 WordNet
    confidence = confs[i] * 100 # 获取置信度
    pred_df = pred_df.append({'Class':class_name, 'Class_ID':label_idx, 'Confidence(%)':confidence, 'WordNet':wordnet}, ignore_index=True) # 预测结果表格添加一行
display(pred_df) # 展示预测结果表格


# In[ ]:




