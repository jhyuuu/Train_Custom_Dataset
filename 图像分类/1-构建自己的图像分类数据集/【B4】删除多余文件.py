#!/usr/bin/env python
# coding: utf-8

# # 删除多余文件
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 2022-8-1

# ## 导入工具包

# In[14]:


import os
import cv2
from tqdm import tqdm


# ## 准备样例数据集

# In[27]:


# 下载测试数据集压缩包
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/dataset_delete_test.zip')


# In[3]:


# 解压
get_ipython().system('unzip dataset_delete_test.zip >> /dev/null')


# ## 删除系统自动生成的多余文件
# 
# 建议在 Linux 系统中运行爬虫、划分训练集测试集代码

# ### 查看待删除的多余文件

# In[15]:


get_ipython().system("find . -iname '__MACOSX'")


# In[16]:


get_ipython().system("find . -iname '.DS_Store'")


# In[17]:


get_ipython().system("find . -iname '.ipynb_checkpoints'")


# In[9]:


'.DS_Store' in os.listdir('dataset_delete_test/芒果')


# ### 删除多余文件

# In[5]:


get_ipython().system("for i in `find . -iname '__MACOSX'`; do rm -rf $i;done")


# In[6]:


get_ipython().system("for i in `find . -iname '.DS_Store'`; do rm -rf $i;done")


# In[7]:


get_ipython().system("for i in `find . -iname '.ipynb_checkpoints'`; do rm -rf $i;done")


# ### 验证多余文件已删除

# In[8]:


get_ipython().system("find . -iname '__MACOSX'")


# In[9]:


get_ipython().system("find . -iname '.DS_Store'")


# In[10]:


get_ipython().system("find . -iname '.ipynb_checkpoints'")


# ## 删除gif格式的图像文件

# In[18]:


dataset_path = 'dataset_delete_test'


# In[19]:


for fruit in tqdm(os.listdir(dataset_path)):
    for file in os.listdir(os.path.join(dataset_path, fruit)):
        file_path = os.path.join(dataset_path, fruit, file)
        img = cv2.imread(file_path)
        if img is None:
            print(file_path, '读取错误，删除')
            os.remove(file_path)


# ## 删除非三通道的图像

# In[67]:


for fruit in tqdm(os.listdir(dataset_path)):
    for file in os.listdir(os.path.join(dataset_path, fruit)):
        file_path = os.path.join(dataset_path, fruit, file)
        img = np.array(Image.open(file_path))
        try:
            channel = img.shape[2]
            if channel != 3:
                print(file_path, '非三通道，删除')
                os.remove(file_path)
        except:
            print(file_path, '非三通道，删除')
            os.remove(file_path)


# ## 再次删除多余的`.ipynb_checkpoints`目录

# In[20]:


get_ipython().system("find . -iname '.ipynb_checkpoints'")


# In[21]:


get_ipython().system("for i in `find . -iname '.ipynb_checkpoints'`; do rm -rf $i;done")


# In[22]:


get_ipython().system("find . -iname '.ipynb_checkpoints'")


# In[ ]:




