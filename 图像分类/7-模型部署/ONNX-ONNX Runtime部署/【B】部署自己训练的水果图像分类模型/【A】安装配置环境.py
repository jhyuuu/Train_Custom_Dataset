#!/usr/bin/env python
# coding: utf-8

# # 安装配置环境
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 
# 2022-8-19

# ## 安装基础工具包

# In[1]:


get_ipython().system('pip install numpy pandas matplotlib tqdm opencv-python pillow onnx onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple')


# ## 安装Pytorch

# In[2]:


get_ipython().system('pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113')


# ## 创建目录

# In[1]:


import os


# In[2]:


# 存放测试图片
os.mkdir('test_img')

# 存放结果文件
os.mkdir('output')

# 存放训练得到的模型权重
os.mkdir('checkpoints')


# In[3]:


# 下载测试图像文件 至 test_img 文件夹

# 草莓图像，来源：https://www.pexels.com/zh-cn/photo/4828489/
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/0818/test_草莓.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_fruits.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_orange_2.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_bananan.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_kiwi.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_石榴.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_orange.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_lemon.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_火龙果.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/watermelon1.jpg -P test_img')

get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/banana1.jpg -P test_img')


# ## 下载中文字体文件

# In[9]:


get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf')


# ## 下载训练好的模型文件

# In[10]:


# 下载样例模型文件
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/checkpoints/fruit30_pytorch_20220814.pth -P checkpoints')


# In[ ]:




