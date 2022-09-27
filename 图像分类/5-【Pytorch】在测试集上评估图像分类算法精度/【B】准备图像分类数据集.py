#!/usr/bin/env python
# coding: utf-8

# # 准备图像分类数据集和模型文件
# 
# 同济子豪兄：https://space.bilibili.com/1900783
# 
# [代码运行云GPU环境](https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1)：GPU RTX 3060、CUDA v11.2
# 
# ## 构建自己的图像分类数据集
# 
# https://www.bilibili.com/video/BV1Jd4y1T7rw

# ## 下载样例数据集

# In[2]:


# 下载数据集压缩包
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/fruit30/fruit30_split.zip')


# In[3]:


# 解压
get_ipython().system('unzip fruit30_split.zip >> /dev/null')
# 删除压缩包
get_ipython().system('rm fruit30_split.zip')


# In[4]:


# 下载 类别名称 和 ID索引号 的映射字典
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/fruit30/idx_to_labels.npy')


# ## 查看数据集目录结构

# In[4]:


get_ipython().system('sudo snap install tree')


# In[7]:


get_ipython().system('tree fruit30_split -L 2')


# ## 训练好的模型文件

# In[5]:


# 下载样例模型文件
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/checkpoints/fruit30_pytorch_20220814.pth -P checkpoints')


# In[ ]:




