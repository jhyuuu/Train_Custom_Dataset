#!/usr/bin/env python
# coding: utf-8

# # 安装配置环境
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行[云GPU平台](https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1)
# 
# 2022-8-7

# ## 直接运行代码块即可

# In[1]:


get_ipython().system('pip install numpy pandas matplotlib seaborn plotly requests tqdm opencv-python pillow wandb -i https://pypi.tuna.tsinghua.edu.cn/simple')


# ## 下载安装Pytorch

# In[5]:


get_ipython().system('pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113')


# ## 下载中文字体文件

# In[1]:


get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf --no-check-certificate')


# ## 创建目录

# In[2]:


import os


# In[3]:


# 存放结果文件
os.mkdir('output')

# 存放训练得到的模型权重
os.mkdir('checkpoints')

# 存放生成的图表
os.mkdir('图表')


# ## 设置matplotlib中文字体

# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# # windows操作系统
# plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签 
# plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号


# In[3]:


# Mac操作系统，参考 https://www.ngui.cc/51cto/show-727683.html
# 下载 simhei.ttf 字体文件
# !wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf


# In[1]:


# Linux操作系统，例如 云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 如果报错 Unable to establish SSL connection.，重新运行本代码块即可
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf -O /environment/miniconda3/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf --no-check-certificate')
get_ipython().system('rm -rf /home/featurize/.cache/matplotlib')

import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rc("font",family='SimHei') # 中文字体
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号


# In[2]:


plt.plot([1,2,3], [100,500,300])
plt.title('matplotlib中文字体测试', fontsize=25)
plt.xlabel('X轴', fontsize=15)
plt.ylabel('Y轴', fontsize=15)
plt.show()


# In[ ]:




