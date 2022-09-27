#!/usr/bin/env python
# coding: utf-8

# # ImageNet预训练图像分类模型预测视频文件
# 
# 使用 ImageNet 预训练图像分类模型，对视频文件执行预测。
# 
# 同济子豪兄：https://space.bilibili.com/1900783
# 
# [代码运行云GPU环境](https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1)：GPU RTX 3060、CUDA v11.2

# ## 导入工具包

# In[28]:


import os
import time
import shutil
import tempfile
from tqdm import tqdm

import cv2
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
import gc

import torch
import torch.nn.functional as F
from torchvision import models

import mmcv

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)


# In[29]:


# 后端绘图，不显示，只保存
import matplotlib
matplotlib.use('Agg')


# ## 载入预训练图像分类模型

# In[30]:


model = models.resnet18(pretrained=True)
model = model.eval()
model = model.to(device)


# ## 载入ImageNet 1000图像分类标签

# In[31]:


df = pd.read_csv('imagenet_class_index.csv')
idx_to_labels = {}
for idx, row in df.iterrows():
    idx_to_labels[row['ID']] = [row['wordnet'], row['class']]


# In[32]:


# idx_to_labels


# ## 图像预处理

# In[33]:


from torchvision import transforms

# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ])


# ## 图像分类预测函数（同上个代码教程）

# In[35]:


def pred_single_frame(img, n=5):
    '''
    输入摄像头画面bgr-array，输出前n个图像分类预测结果的图像bgr-array
    '''
    img_bgr = img
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR 转 RGB
    img_pil = Image.fromarray(img_rgb) # array 转 pil
    input_img = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
    pred_logits = model(input_img) # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算
    
    top_n = torch.topk(pred_softmax, n) # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze() # 解析出类别
    confs = top_n[0].cpu().detach().numpy().squeeze() # 解析出置信度
    
    # 在图像上写字
    for i in range(n):
        class_name = idx_to_labels[pred_ids[i]][1] # 获取类别名称
        confidence = confs[i] * 100 # 获取置信度
        text = '{:<15} {:>.4f}'.format(class_name, confidence)

        # !图片，添加的文字，左上角坐标，字体，字号，bgr颜色，线宽
        img_bgr = cv2.putText(img_bgr, text, (25, 50 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)
        
    return img_bgr, pred_softmax


# ## 视频预测

# ### 输入输出视频路径

# In[36]:


input_video = 'test_img/video_3.mp4'


# ### 可视化方案一：原始图像+预测结果文字

# In[37]:


# 创建临时文件夹，存放每帧结果
temp_out_dir = time.strftime('%Y%m%d%H%M%S')
os.mkdir(temp_out_dir)
print('创建文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))


# In[38]:


# 读入待预测视频
imgs = mmcv.VideoReader(input_video)

prog_bar = mmcv.ProgressBar(len(imgs))

# 对视频逐帧处理
for frame_id, img in enumerate(imgs):
    
    ## 处理单帧画面
    img, pred_softmax = pred_single_frame(img, n=5)

    # 将处理后的该帧画面图像文件，保存至 /tmp 目录下
    cv2.imwrite(f'{temp_out_dir}/{frame_id:06d}.jpg', img)
    
    prog_bar.update() # 更新进度条

# 把每一帧串成视频文件
mmcv.frames2video(temp_out_dir, 'output/output_pred.mp4', fps=imgs.fps, fourcc='mp4v')

shutil.rmtree(temp_out_dir) # 删除存放每帧画面的临时文件夹
print('删除临时文件夹', temp_out_dir)


# ### 可视化方案二：原始图像+预测结果文字+各类别置信度柱状图

# In[39]:


def pred_single_frame_bar(img):
    '''
    输入pred_single_frame函数输出的bgr-array，加柱状图，保存
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR 转 RGB
    fig = plt.figure(figsize=(18,6))
    # 绘制左图-视频图
    ax1 = plt.subplot(1,2,1)
    ax1.imshow(img)
    ax1.axis('off')
    # 绘制右图-柱状图
    ax2 = plt.subplot(1,2,2)
    x = range(1000)
    y = pred_softmax.cpu().detach().numpy()[0]
    ax2.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
    plt.xlabel('类别', fontsize=20)
    plt.ylabel('置信度', fontsize=20)
    ax2.tick_params(labelsize=16) # 坐标文字大小
    plt.ylim([0, 1.0]) # y轴取值范围
    plt.xlabel('类别',fontsize=25)
    plt.ylabel('置信度',fontsize=25)
    plt.title('图像分类预测结果', fontsize=30)
    
    plt.tight_layout()
    fig.savefig(f'{temp_out_dir}/{frame_id:06d}.jpg')
    # 释放内存
    fig.clf()
    plt.close()
    gc.collect()


# In[40]:


# 创建临时文件夹，存放每帧结果
temp_out_dir = time.strftime('%Y%m%d%H%M%S')
os.mkdir(temp_out_dir)
print('创建文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))


# In[41]:


# 读入待预测视频
imgs = mmcv.VideoReader(input_video)

prog_bar = mmcv.ProgressBar(len(imgs))

# 对视频逐帧处理
for frame_id, img in enumerate(imgs):
    
    ## 处理单帧画面
    img, pred_softmax = pred_single_frame(img, n=5)
    img = pred_single_frame_bar(img)
    
    prog_bar.update() # 更新进度条

# 把每一帧串成视频文件
mmcv.frames2video(temp_out_dir, 'output/output_bar.mp4', fps=imgs.fps, fourcc='mp4v')

shutil.rmtree(temp_out_dir) # 删除存放每帧画面的临时文件夹
print('删除临时文件夹', temp_out_dir)


# In[ ]:




