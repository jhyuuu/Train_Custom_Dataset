#!/usr/bin/env python
# coding: utf-8

# # 导出ONNX模型
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# 代码运行云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 
# 2022-8-22

# ## 导入工具包

# In[1]:


import torch
from torchvision import models

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


# ## 导入训练好的模型

# In[3]:


model = torch.load('checkpoints/fruit30_pytorch_20220814.pth')
model = model.eval().to(device)


# In[4]:


x = torch.randn(1, 3, 256, 256).to(device)


# In[5]:


output = model(x)


# In[6]:


output.shape


# ## Pytorch模型转ONNX模型

# In[8]:


x = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    torch.onnx.export(
        model,                   # 要转换的模型
        x,                       # 模型的任意一组输入
        'fruit30_resnet18.onnx', # 导出的 ONNX 文件名
        opset_version=11,        # ONNX 算子集版本
        input_names=['input'],   # 输入 Tensor 的名称（自己起名字）
        output_names=['output']  # 输出 Tensor 的名称（自己起名字）
    ) 


# ## 验证onnx模型导出成功

# In[9]:


import onnx

# 读取 ONNX 模型
onnx_model = onnx.load('fruit30_resnet18.onnx')

# 检查模型格式是否正确
onnx.checker.check_model(onnx_model)

print('无报错，onnx模型载入成功')


# ## 以可读的形式打印计算图

# In[10]:


print(onnx.helper.printable_graph(onnx_model.graph))


# ## 使用Netron对onnx模型可视化
# 
# https://netron.app

# In[ ]:




