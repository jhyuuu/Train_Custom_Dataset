#!/usr/bin/env python
# coding: utf-8

# # 测试集语义特征UMAP降维可视化
# 
# 抽取Pytorch训练得到的图像分类模型中间层的输出特征，作为输入图像的语义特征。
# 
# 计算测试集所有图像的语义特征，使用t-SNE和UMAP两种降维方法降维至二维和三维，可视化。
# 
# 分析不同类别的语义距离、异常数据、细粒度分类、高维数据结构。
# 
# 同济子豪兄：https://space.bilibili.com/1900783
# 
# [代码运行云GPU环境](https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1)：GPU RTX 3060、CUDA v11.2
# 
# 参考文档：https://umap-learn.readthedocs.io/en/latest/plotting.html

# ## 安装UMAP

# In[2]:


# 官方文档：https://umap-learn.readthedocs.io/en/latest/index.html
get_ipython().system('pip install umap-learn datashader bokeh holoviews scikit-image colorcet')


# ## 设置matplotlib中文字体

# In[1]:


# # windows操作系统
# plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签 
# plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号


# In[2]:


# Mac操作系统，参考 https://www.ngui.cc/51cto/show-727683.html
# 下载 simhei.ttf 字体文件
# !wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf


# In[3]:


# Linux操作系统，例如 云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 如果遇到 SSL 相关报错，重新运行本代码块即可
get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf -O /environment/miniconda3/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf')
get_ipython().system('rm -rf /home/featurize/.cache/matplotlib')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font",family='SimHei') # 中文字体
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号


# In[3]:


plt.plot([1,2,3], [100,500,300])
plt.title('matplotlib中文字体测试', fontsize=25)
plt.xlabel('X轴', fontsize=15)
plt.ylabel('Y轴', fontsize=15)
plt.show()


# ## 导入工具包

# In[4]:


import numpy as np
import pandas as pd
import cv2


# ## 载入测试集图像语义特征

# In[130]:


encoding_array = np.load('测试集语义特征.npy', allow_pickle=True)


# In[131]:


encoding_array.shape


# ## 载入测试集图像分类结果

# In[132]:


df = pd.read_csv('测试集预测结果.csv')


# In[133]:


df.head()


# In[134]:


classes = df['标注类别名称'].unique()
print(classes)


# ## 可视化配置

# In[135]:


import seaborn as sns
marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


# In[136]:


class_list = np.unique(df['标注类别名称'])


# In[137]:


class_list


# In[138]:


n_class = len(class_list) # 测试集标签类别数
palette = sns.hls_palette(n_class) # 配色方案
sns.palplot(palette)


# In[139]:


# 随机打乱颜色列表和点型列表
import random
random.seed(1234)
random.shuffle(marker_list)
random.shuffle(palette)


# ## UMAP降维至二维可视化

# In[140]:


import umap
import umap.plot


# In[141]:


mapper = umap.UMAP(n_neighbors=10, n_components=2, random_state=12).fit(encoding_array)


# In[142]:


mapper.embedding_.shape


# In[143]:


X_umap_2d = mapper.embedding_


# In[144]:


X_umap_2d.shape


# In[145]:


# 不同的 符号 表示 不同的 标注类别
show_feature = '标注类别名称'


# In[146]:


plt.figure(figsize=(14, 14))
for idx, fruit in enumerate(class_list): # 遍历每个类别
    # 获取颜色和点型
    color = palette[idx]
    marker = marker_list[idx%len(marker_list)]

    # 找到所有标注类别为当前类别的图像索引号
    indices = np.where(df[show_feature]==fruit)
    plt.scatter(X_umap_2d[indices, 0], X_umap_2d[indices, 1], color=color, marker=marker, label=fruit, s=150)

plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
plt.xticks([])
plt.yticks([])
plt.savefig('语义特征UMAP二维降维可视化.pdf', dpi=300) # 保存图像
plt.show()


# ## 来了一张新图像，可视化语义特征

# 下载新图像

# In[147]:


get_ipython().system('wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/0818/test_kiwi.jpg')


# 导入模型、预处理

# In[149]:


import cv2
import torch
from PIL import Image
from torchvision import transforms

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load('checkpoints/fruit30_pytorch_20220814.pth')
model = model.eval().to(device)

from torchvision.models.feature_extraction import create_feature_extractor
model_trunc = create_feature_extractor(model, return_nodes={'avgpool': 'semantic_feature'})

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ])


# 计算新图像的语义特征

# In[150]:


img_path = 'test_kiwi.jpg'
img_pil = Image.open(img_path)
input_img = test_transform(img_pil) # 预处理
input_img = input_img.unsqueeze(0).to(device)
# 执行前向预测，得到指定中间层的输出
pred_logits = model_trunc(input_img)
semantic_feature = pred_logits['semantic_feature'].squeeze().detach().cpu().numpy().reshape(1,-1)


# In[151]:


semantic_feature.shape


# 对新图像语义特征降维

# In[152]:


# umap降维
new_embedding = mapper.transform(semantic_feature)[0]


# In[153]:


new_embedding


# In[154]:


plt.figure(figsize=(14, 14))
for idx, fruit in enumerate(class_list): # 遍历每个类别
    # 获取颜色和点型
    color = palette[idx]
    marker = marker_list[idx%len(marker_list)]

    # 找到所有标注类别为当前类别的图像索引号
    indices = np.where(df[show_feature]==fruit)
    plt.scatter(X_umap_2d[indices, 0], X_umap_2d[indices, 1], color=color, marker=marker, label=fruit, s=150)

plt.scatter(new_embedding[0], new_embedding[1], color='r', marker='X', label=img_path, s=1000)

plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
plt.xticks([])
plt.yticks([])
plt.savefig('语义特征UMAP二维降维可视化-新图像.pdf', dpi=300) # 保存图像
plt.show()


# ## plotply交互式可视化

# In[155]:


import plotly.express as px


# In[156]:


df_2d = pd.DataFrame()
df_2d['X'] = list(X_umap_2d[:, 0].squeeze())
df_2d['Y'] = list(X_umap_2d[:, 1].squeeze())
df_2d['标注类别名称'] = df['标注类别名称']
df_2d['预测类别'] = df['top-1-预测名称']
df_2d['图像路径'] = df['图像路径']
df_2d.to_csv('UMAP-2D.csv', index=False)


# In[157]:


# 增加新图像的一行
new_img_row = {
    'X':new_embedding[0],
    'Y':new_embedding[1],
    '标注类别名称':img_path,
    '图像路径':img_path
}

df_2d = df_2d.append(new_img_row, ignore_index=True)


# In[158]:


df_2d


# In[159]:


fig = px.scatter(df_2d, 
                 x='X', 
                 y='Y',
                 color=show_feature, 
                 labels=show_feature,
                 symbol=show_feature, 
                 hover_name='图像路径',
                 opacity=0.8,
                 width=1000, 
                 height=600
                )
# 设置排版
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_html('语义特征UMAP二维降维plotly可视化.html')


# In[27]:


# 查看图像
img_path_temp = 'fruit30_split/val/火龙果/3.jpg'
img_bgr = cv2.imread(img_path_temp)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
temp_df = df[df['图像路径'] == img_path_temp]
title_str = img_path_temp + '\nTrue:' + temp_df['标注类别名称'].item() + ' Pred:' + temp_df['top-1-预测名称'].item()
plt.title(title_str)
plt.show()


# ## UMAP降维至三维，并可视化

# In[160]:


mapper = umap.UMAP(n_neighbors=10, n_components=3, random_state=12).fit(encoding_array)


# In[161]:


X_umap_3d = mapper.embedding_


# In[162]:


X_umap_3d.shape


# In[163]:


show_feature = '标注类别名称'
# show_feature = '预测类别'


# In[164]:


df_3d = pd.DataFrame()
df_3d['X'] = list(X_umap_3d[:, 0].squeeze())
df_3d['Y'] = list(X_umap_3d[:, 1].squeeze())
df_3d['Z'] = list(X_umap_3d[:, 2].squeeze())
df_3d['标注类别名称'] = df['标注类别名称']
df_3d['预测类别'] = df['top-1-预测名称']
df_3d['图像路径'] = df['图像路径']
df_3d.to_csv('UMAP-3D.csv', index=False)


# In[165]:


df_3d


# In[166]:


fig = px.scatter_3d(df_3d, 
                    x='X', 
                    y='Y', 
                    z='Z',
                    color=show_feature, 
                    labels=show_feature,
                    symbol=show_feature, 
                    hover_name='图像路径',
                    opacity=0.6,
                    width=1000, 
                    height=800)

# 设置排版
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_html('语义特征UMAP三维降维plotly可视化.html')


# ## 来了一张新图像，可视化语义特征

# In[167]:


# umap降维
new_embedding = mapper.transform(semantic_feature)[0]


# In[168]:


# 增加新图像的一行
new_img_row = {
    'X':new_embedding[0],
    'Y':new_embedding[1],
    'Z':new_embedding[2],
    '标注类别名称':img_path,
    '图像路径':img_path
}

df_3d = df_3d.append(new_img_row, ignore_index=True)


# In[169]:


df_3d


# In[170]:


fig = px.scatter_3d(df_3d, 
                    x='X', 
                    y='Y', 
                    z='Z',
                    color=show_feature, 
                    labels=show_feature,
                    symbol=show_feature, 
                    hover_name='图像路径',
                    opacity=0.6,
                    width=1000, 
                    height=800)

# 设置排版
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_html('语义特征UMAP三维降维plotly可视化.html')


# In[ ]:




