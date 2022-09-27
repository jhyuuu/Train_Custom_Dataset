#!/usr/bin/env python
# coding: utf-8

# # 安装配置环境
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 
# 2022-8-22

# ## 安装Pytorch

# In[2]:


get_ipython().system('pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113')


# ## 安装工具包

# In[1]:


get_ipython().system('pip install numpy pandas matplotlib tqdm opencv-python pillow onnx onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[ ]:




