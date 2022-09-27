#!/usr/bin/env python
# coding: utf-8

# # ImageNet预训练图像分类模型预测单张图像-英文
# 
# 使用 ImageNet 预训练图像分类模型，对单张图像文件执行前向预测。
# 
# 同济子豪兄：https://space.bilibili.com/1900783
# 
# [代码运行云GPU环境](https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1)：GPU RTX 3060、CUDA v11.2

# ## 导入基础工具包

# In[1]:


import os

import cv2

import pandas as pd
import numpy as np

import torch

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 计算设备

# In[2]:


# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[3]:


print('device', device)


# ## 载入预训练图像分类模型
# 
# 子豪兄精读AI论文视频合集：https://space.bilibili.com/1900783/channel/seriesdetail?sid=250032

# In[4]:


from torchvision import models


# In[5]:


# 载入预训练图像分类模型

model = models.resnet18(pretrained=True) 

# model = models.resnet152(pretrained=True)


# In[6]:


model = model.eval()
model = model.to(device)


# ## 图像预处理

# In[7]:


from torchvision import transforms

# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ])


# ## 载入一张测试图像

# In[10]:


# img_path = 'test_img/banana1.jpg'
# img_path = 'test_img/husky1.jpeg'
# img_path = 'test_img/basketball_shoe.jpeg'
img_path = './basketball_shoe.png'

# img_path = 'test_img/cat_dog.jpg'


# In[11]:


# 用 pillow 载入
from PIL import Image
img_pil = Image.open(img_path)


# In[10]:


img_pil


# In[11]:


np.array(img_pil).shape


# ## 执行图像分类预测

# In[12]:


input_img = test_transform(img_pil) # 预处理


# In[13]:


input_img.shape


# In[14]:


input_img = input_img.unsqueeze(0).to(device)


# In[15]:


input_img.shape


# In[16]:


# 执行前向预测，得到所有类别的 logit 预测分数
pred_logits = model(input_img) 


# In[17]:


pred_logits.shape


# In[18]:


# pred_logits


# In[19]:


import torch.nn.functional as F
pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算


# In[20]:


pred_softmax.shape


# In[21]:


# pred_softmax


# ## 预测结果分析

# ### 各类别置信度柱状图

# In[22]:


plt.figure(figsize=(8,4))

x = range(1000)
y = pred_softmax.cpu().detach().numpy()[0]

ax = plt.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
plt.ylim([0, 1.0]) # y轴取值范围
# plt.bar_label(ax, fmt='%.2f', fontsize=15) # 置信度数值

plt.xlabel('Class', fontsize=20)
plt.ylabel('Confidence', fontsize=20)
plt.tick_params(labelsize=16) # 坐标文字大小
plt.title(img_path, fontsize=25)

plt.show()


# ### 取置信度最大的 n 个结果

# In[23]:


n = 10
top_n = torch.topk(pred_softmax, n)


# In[24]:


top_n


# In[25]:


# 解析出类别
pred_ids = top_n[1].cpu().detach().numpy().squeeze()


# In[26]:


pred_ids


# In[27]:


# 解析出置信度
confs = top_n[0].cpu().detach().numpy().squeeze()


# In[28]:


confs


# ### 载入ImageNet 1000图像分类标签
# 
# ImageNet 1000类别中文释义：https://github.com/ningbonb/imagenet_classes_chinese

# In[29]:


df = pd.read_csv('imagenet_class_index.csv')


# In[30]:


df


# In[31]:


idx_to_labels = {}
for idx, row in df.iterrows():
    idx_to_labels[row['ID']] = [row['wordnet'], row['class']]


# In[32]:


# idx_to_labels


# ### 图像分类结果写在原图上

# In[33]:


# 用 opencv 载入原图
img_bgr = cv2.imread(img_path)


# In[34]:


for i in range(n):
    class_name = idx_to_labels[pred_ids[i]][1] # 获取类别名称
    confidence = confs[i] * 100 # 获取置信度
    text = '{:<15} {:>.4f}'.format(class_name, confidence)
    print(text)
    
    # !图片，添加的文字，左上角坐标，字体，字号，bgr颜色，线宽
    img_bgr = cv2.putText(img_bgr, text, (25, 50 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)


# > 注意，ImageNet 1000类中并不包含“西瓜”

# In[35]:


# 保存图像
cv2.imwrite('output/img_pred.jpg', img_bgr)


# In[36]:


# 载入预测结果图像
img_pred = Image.open('output/img_pred.jpg')
img_pred


# ### 图像和柱状图一起显示

# In[37]:


fig = plt.figure(figsize=(18,6))

# 绘制左图-预测图
ax1 = plt.subplot(1,2,1)
ax1.imshow(img_pred)
ax1.axis('off')

# 绘制右图-柱状图
ax2 = plt.subplot(1,2,2)
x = df['ID']
y = pred_softmax.cpu().detach().numpy()[0]
ax2.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)

plt.ylim([0, 1.0]) # y轴取值范围
plt.title('{} Classification'.format(img_path), fontsize=30)
plt.xlabel('Class', fontsize=20)
plt.ylabel('Confidence', fontsize=20)
ax2.tick_params(labelsize=16) # 坐标文字大小

plt.tight_layout()
fig.savefig('output/预测图+柱状图.jpg')


# ### 预测结果表格输出

# In[38]:


pred_df = pd.DataFrame() # 预测结果表格
for i in range(n):
    class_name = idx_to_labels[pred_ids[i]][1] # 获取类别名称
    label_idx = int(pred_ids[i]) # 获取类别号
    wordnet = idx_to_labels[pred_ids[i]][0] # 获取 WordNet
    confidence = confs[i] * 100 # 获取置信度
    pred_df = pred_df.append({'Class':class_name, 'Class_ID':label_idx, 'Confidence(%)':confidence, 'WordNet':wordnet}, ignore_index=True) # 预测结果表格添加一行
display(pred_df) # 展示预测结果表格


# In[ ]:





# In[ ]:




