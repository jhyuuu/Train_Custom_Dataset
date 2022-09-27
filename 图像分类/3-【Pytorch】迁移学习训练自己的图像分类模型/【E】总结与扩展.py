#!/usr/bin/env python
# coding: utf-8

# # 总结与扩展
# 
# 同济子豪兄：https://space.bilibili.com/1900783

# ## 注意事项
# 
# 严禁把测试集图像用于训练（反向传播更新权重）
# 
# 抛开baseline基准模型谈性能（速度、精度），都是耍流氓
# 
# 测试集上的准确率越高，模型就一定越好吗？
# 
# 常用数据集中存在大量的错标、漏标：https://mp.weixin.qq.com/s/4NbIA4wsNdX-N2uMOUmPLA
# 

# ## 创新点展望
# 
# 更换不同预训练图像分类模型
# 
# 分别尝试三种不同的迁移学习训练配置：只微调训练模型最后一层（全连接分类层）、微调训练所有层、随机初始化模型全部权重，从头训练所有层
# 
# 更换不同的优化器、学习率

# ## 扩展阅读
# 
# 同济子豪兄的论文精读视频：https://openmmlab.feishu.cn/docs/doccnWv17i1svV19T0QquS0gKFc
# 
# 开源图像分类算法库 MMClassificaiton：https://github.com/open-mmlab/mmclassification
# 
# **机器学习分类评估指标**
# 
# 公众号 人工智能小技巧 回复 混淆矩阵
# 
# 手绘笔记讲解：https://www.bilibili.com/video/BV1iJ41127wr?p=3
# 
# 混淆矩阵：
# https://www.bilibili.com/video/BV1iJ41127wr?p=4
# 
# https://www.bilibili.com/video/BV1iJ41127wr?p=5
# 
# ROC曲线：
# https://www.bilibili.com/video/BV1iJ41127wr?p=6
# 
# https://www.bilibili.com/video/BV1iJ41127wr?p=7
# 
# https://www.bilibili.com/video/BV1iJ41127wr?p=8
# 
# F1-score：https://www.bilibili.com/video/BV1iJ41127wr?p=9
# 
# F-beta-score：https://www.bilibili.com/video/BV1iJ41127wr?p=10

# ## 训练好图像分类模型之后，做什么？
# 
# 在新图像、视频、摄像头实时画面预测
# 
# 在测试集上评估：混淆矩阵、ROC曲线、PR曲线、语义特征降维可视化
# 
# 可解释性分析：CAM热力图
# 
# 模型TensorRT部署：智能手机、开发板、浏览器、服务器
# 
# 转ONNX并可视化模型结构
# 
# MMClassification图像分类、MMDeploy模型部署
# 
# 开发图像分类APP和微信小程序

# In[ ]:




