#!/usr/bin/env python
# coding: utf-8

# # ImageNet预训练图像分类模型预测摄像头实时画面
# 
# 使用 ImageNet 预训练图像分类模型，预测摄像头实时画面。
# 
# 本代码需在`连接摄像头的本地`运行，不能在云GPU平台运行。
# 
# 同济子豪兄：https://space.bilibili.com/1900783
# 
# 测试运行环境：Macbook Pro

# ## 导入工具包

# In[1]:


import os

import numpy as np
import pandas as pd

import cv2 # opencv-python
from PIL import Image # pillow
from tqdm import tqdm # 进度条

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn.functional as F
from torchvision import models


# In[2]:


# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)


# ## 载入预训练图像分类模型

# In[13]:


model = models.resnet18(pretrained=True)
model = model.eval()
model = model.to(device)


# ## 载入ImageNet 1000图像分类标签
# 
# ImageNet 1000类别中文释义：https://github.com/ningbonb/imagenet_classes_chinese

# In[14]:


df = pd.read_csv('imagenet_class_index.csv')
idx_to_labels = {}
for idx, row in df.iterrows():
    idx_to_labels[row['ID']] = row['class']


# In[15]:


# idx_to_labels


# ## 图像预处理

# In[16]:


from torchvision import transforms

# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ])


# ## 获取摄像头的一帧画面

# In[9]:


# 导入opencv-python
import cv2
import time

# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(1)

# 打开cap
cap.open(0)

time.sleep(1)

success, img_bgr = cap.read()
    
# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()


# In[10]:


img_bgr.shape


# In[11]:


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # BGR转RGB


# In[12]:


img_pil = Image.fromarray(img_rgb)


# In[13]:


img_pil


# In[14]:


input_img = test_transform(img_pil).unsqueeze(0).to(device) # 预处理


# In[15]:


input_img.shape


# In[16]:


pred_logits = model(input_img) # 执行前向预测，得到所有类别的 logit 预测分数
pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算


# In[17]:


n = 5


# In[18]:


top_n = torch.topk(pred_softmax, n) # 取置信度最大的 n 个结果


# In[19]:


confs = top_n[0].cpu().detach().numpy().squeeze()


# In[20]:


confs


# In[21]:


pred_ids = top_n[1].cpu().detach().numpy().squeeze()


# In[22]:


pred_ids


# In[23]:


for i in range(len(confs)):
    pred_class = idx_to_labels[pred_ids[i]]
    text = '{:<15} {:>.3f}'.format(pred_class, confs[i])
    
    # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，线宽，线型
    img_bgr = cv2.putText(img_bgr, text, (50, 80 + 80 * i), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5, cv2.LINE_AA)
    


# In[25]:


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # BGR转RGB
plt.imshow(img_rgb)
plt.show()


# ## 处理单帧画面的函数（英文）

# In[17]:


# 处理帧函数
def process_frame(img):
    
    '''
    输入摄像头拍摄画面bgr-array，输出图像分类预测结果bgr-array
    '''
    
    # 记录该帧开始处理的时间
    start_time = time.time()
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR转RGB
    img_pil = Image.fromarray(img_rgb) # array 转 PIL
    input_img = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
    pred_logits = model(input_img) # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算
    
    top_n = torch.topk(pred_softmax, 5) # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze() # 解析预测类别
    confs = top_n[0].cpu().detach().numpy().squeeze() # 解析置信度
    
    # 在图像上写字
    for i in range(len(confs)):
        pred_class = idx_to_labels[pred_ids[i]]
        text = '{:<15} {:>.3f}'.format(pred_class, confs[i])

        # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，线宽，线型
        img = cv2.putText(img, text, (50, 160 + 80 * i), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
    
    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1/(end_time - start_time)  
    # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，线宽，线型
    img = cv2.putText(img, 'FPS  '+str(int(FPS)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4, cv2.LINE_AA)

    return img


# ## 调用摄像头获取每帧（模板）

# In[18]:


# 调用摄像头逐帧实时处理模板
# 不需修改任何代码，只需修改process_frame函数即可
# 同济子豪兄 2021-7-8

# 导入opencv-python
import cv2
import time

# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(1)

# 打开cap
cap.open(0)

# 无限循环，直到break被触发
while cap.isOpened():
    # 获取画面
    success, frame = cap.read()
    if not success:
        print('Error')
        break
    
    ## !!!处理帧函数
    frame = process_frame(frame)
    
    # 展示处理后的三通道图像
    cv2.imshow('my_window',frame)

    if cv2.waitKey(1) in [ord('q'),27]: # 按键盘上的q或esc退出（在英文输入法下）
        break
    
# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()


# In[ ]:




