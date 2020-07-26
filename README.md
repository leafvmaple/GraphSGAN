# GraphSGAN

## 细节

复现过程主要采用pytorch来实现，部分细节参考自论文源码

## 改进

时间仓促，未能对模型进行有效的改进

## 得分

Accurracy: 80.2
Recall: 50.5

与原论文存在一定差异，大致与Embedding-Planetoid相当

## 问题

1. 时间仓促，参数迭代不足应该是与原文相差过大的很重要原因
2. Upper bound也未能实现，导致Recall相差比较大。

## 参考

[Github](https://github.com/THUDM/GraphSGAN)