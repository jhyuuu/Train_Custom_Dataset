#!/usr/bin/env python
# coding: utf-8

# # 安装配置环境
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行[云GPU平台](https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1)
# 
# 2022-8-2

# ## 直接运行代码块即可

# In[ ]:


get_ipython().system('pip install numpy pandas matplotlib requests tqdm opencv-python pillow gc -i https://pypi.tuna.tsinghua.edu.cn/simple')


# ## 下载安装Pytorch

# In[1]:


get_ipython().system('pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113')


# ## 下载安装 mmcv-full

# In[2]:


# 安装mmcv -full
get_ipython().system('pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html')


# ## 下载中文字体文件

# In[ ]:


get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf')


# ## 下载 ImageNet 1000类别信息

# In[ ]:


get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/meta_data/imagenet_class_index.csv')


# ## 创建目录

# In[3]:


import os


# In[ ]:


# 存放测试图片
os.mkdir('test_img')

# 存放结果文件
os.mkdir('output')


# In[ ]:


# 下载测试图像文件 至 test_img 文件夹

get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/watermelon1.jpg -O test_img/watermelon1.jpg')
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/banana1.jpg -O test_img/banana1.jpg')
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/cat1.jpg -O test_img/cat1.jpg')

# 哈士奇，来源：https://www.pexels.com/zh-cn/photo/2853130/
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/husky1.jpeg -O test_img/husky1.jpeg')

# 猫狗，来源：https://unsplash.com/photos/ouo1hbizWwo
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/cat_dog.jpg -O test_img/cat_dog.jpg')

get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/video_2.mp4 -O test_img/video_2.mp4')


# In[ ]:




