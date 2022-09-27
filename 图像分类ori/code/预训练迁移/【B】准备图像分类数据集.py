#!/usr/bin/env python
# coding: utf-8

# # 准备图像分类数据集
# 
# 同济子豪兄：https://space.bilibili.com/1900783
# 
# [代码运行云GPU环境](https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1)：GPU RTX 3060、CUDA v11.2
# 
# ## 构建自己的图像分类数据集
# 
# https://www.bilibili.com/video/BV1Jd4y1T7rw

# ## 下载样例数据集

# In[1]:


# 下载数据集压缩包
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/fruit30/fruit30_split.zip')


# In[2]:


# 解压
get_ipython().system('unzip fruit30_split.zip >> /dev/null')


# In[3]:


# 删除压缩包
get_ipython().system('rm fruit30_split.zip')


# ## 查看数据集目录结构

# In[19]:


get_ipython().system('sudo snap install tree')


# In[18]:


get_ipython().system('tree fruit30_split -L 2')


# In[ ]:




