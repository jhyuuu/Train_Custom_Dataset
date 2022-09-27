#!/usr/bin/env python
# coding: utf-8

# # 总结与扩展
# 
# 同济子豪兄 https://space.bilibili.com/1900783
# 
# ## 机器学习分类评估指标
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
# 
# ## 语义特征降维可视化
# 
# 【斯坦福CS231N】可视化卷积神经网络：https://www.bilibili.com/video/BV1K7411W7So
# 
# 五万张ImageNet 验证集图像的语义特征降维可视化：https://cs.stanford.edu/people/karpathy/cnnembed/
# 
# 谷歌可视化降维工具Embedding Projector https://www.bilibili.com/video/BV1iJ41127wr?p=11
# 
# ## 思考题
# 
# - 语义特征图的符号，应该显示标注类别，还是预测类别？
# 
# - 如果同一个类别，语义特征降维可视化后却有两个或多个聚类簇，可能原因是什么？（胡萝卜、胡萝卜片、胡萝卜丝）
# 
# - 如果一个类别里混入了另一个类别的图像，可能原因是什么？（标注错误、图像中包含多类、太像、细粒度分类）
# 
# - 语义特征可以取训练集图像计算得到吗？为什么？
# 
# - 取由浅至深不同中间层的输出结果作为语义特征，效果会有何不同？
# 
# - ”神经网络的强大之处，就在于，输入无比复杂的图像像素分布，输出在高维度线性可分的语义特征分布，最后一层分类层就是线性分类器"。如何理解这句话？
# 
# - 如何进一步改进语义特征的可视化？（交互点击显示图片）
# 
# - 如何做出扩展阅读里第二个链接的效果？
# 
# - 对视频或者摄像头实时画面，绘制语义特征降维点的轨迹

# In[ ]:




