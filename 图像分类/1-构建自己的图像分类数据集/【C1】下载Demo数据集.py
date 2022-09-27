#!/usr/bin/env python
# coding: utf-8

# # 下载Demo数据集
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行[云GPU平台](https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1)
# 
# 2022-8-2

# ## melon17瓜果图像分类数据集

# In[ ]:


# 下载压缩包
# 如报错 Unable to establish SSL connection. 重新运行本代码块即可
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/melon17/melon17_full.zip')


# In[ ]:


# 解压
get_ipython().system('unzip melon17_full.zip >> /dev/null')


# ## fruit81水果图像分类数据集

# In[7]:


# 下载压缩包
# 如报错 Unable to establish SSL connection. 重新运行本代码块即可
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/fruit81/fruit81_full.zip')


# In[8]:


# 解压
get_ipython().system('unzip fruit81_full.zip >> /dev/null')


# ## 如何删除文件和文件夹

# In[ ]:


# 删除文件
get_ipython().system('rm -rf fruit81_full.zip')


# In[ ]:


# 删除文件夹
get_ipython().system('rm -rf fruit81_split')


# In[ ]:




