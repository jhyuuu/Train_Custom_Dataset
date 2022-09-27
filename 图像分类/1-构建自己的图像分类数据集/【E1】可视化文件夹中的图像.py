#!/usr/bin/env python
# coding: utf-8

# # 可视化文件夹中的图像
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行[云GPU平台](https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1)
# 
# 2022-7-31

# ## 导入工具包

# In[26]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import math
import os

import cv2

from tqdm import tqdm


# ## 指定要可视化图像的文件夹

# In[27]:


folder_path = 'fruit81_split/train/西瓜'


# In[28]:


# 可视化图像的个数
N = 36


# In[29]:


# n 行 n 列
n = math.floor(np.sqrt(N))
n


# ## 读取文件夹中的所有图像

# In[30]:


images = []
for each_img in os.listdir(folder_path)[:N]:
    img_path = os.path.join(folder_path, each_img)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    images.append(img_rgb)


# In[31]:


len(images)


# ## 画图

# In[32]:


fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig, 111,  # 类似绘制子图 subplot(111)
                 nrows_ncols=(n, n),  # 创建 n 行 m 列的 axes 网格
                 axes_pad=0.02,  # 网格间距
                 share_all=True
                 )

# 遍历每张图像
for ax, im in zip(grid, images):
    ax.imshow(im)
    ax.axis('off')

plt.tight_layout()
plt.show()


# In[ ]:




