#!/usr/bin/env python
# coding: utf-8

# # 导出ONNX模型
# 
# 把原生Pytorch训练得到的图像分类模型，导出为ONNX格式，用于后续在ONNX Runtime推理引擎上部署。
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


# ## 载入ImageNet预训练图像分类模型

# In[2]:


model = models.resnet18(pretrained=True)
model = model.eval().to(device)


# In[3]:


x = torch.randn(1, 3, 256, 256).to(device)


# In[4]:


output = model(x)


# In[5]:


output.shape


# ## Pytorch模型转ONNX模型

# In[6]:


x = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    torch.onnx.export(
        model,                  # 要转换的模型
        x,                      # 模型的任意一组输入
        'resnet18.onnx',        # 导出的 ONNX 文件名
        opset_version=11,       # ONNX 算子集版本
        input_names=['input'],  # 输入 Tensor 的名称（自己起名字）
        output_names=['output'] # 输出 Tensor 的名称（自己起名字）
    ) 


# ## 验证onnx模型导出成功

# In[7]:


import onnx

# 读取 ONNX 模型
onnx_model = onnx.load('resnet18.onnx')

# 检查模型格式是否正确
onnx.checker.check_model(onnx_model)

print('无报错，onnx模型载入成功')


# ## 以可读的形式打印计算图

# In[8]:


print(onnx.helper.printable_graph(onnx_model.graph))


# ## 使用Netron对onnx模型可视化
# 
# https://netron.app

# In[ ]:




