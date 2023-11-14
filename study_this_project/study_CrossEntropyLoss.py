import torch.nn
import torch

loss = torch.nn.CrossEntropyLoss(reduction='mean')

pre = torch.tensor([[0.8, 0.5, 0.2, 0.5],
                    [0.2, 0.9, 0.3, 0.2],
                    [0.4, 0.3, 0.7, 0.1],
                    [0.1, 0.2, 0.4, 0.8]], dtype=torch.float)
tgt2 = torch.tensor([[1, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=torch.float)

tgt = torch.tensor([0, 1, 2, 4], dtype=torch.long)
l = loss(pre, tgt)
l2 = loss(pre, tgt2)
print(l,l2)

import torch
import torch.nn as nn

# 模拟数据
batch_size = 4
num_classes = 3

# 随机生成模型的输出（预测值），形状为 (batch_size, num_classes)
# 这里使用随机数生成，实际中应该是模型的输出
predictions = torch.randn(batch_size, num_classes)

# 随机生成标签（真值），形状为 (batch_size)
# 这里使用随机数生成，实际中应该是真实标签
labels = torch.randint(0, num_classes, (batch_size,))

# 创建交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 计算损失
loss = criterion(predictions, labels)
print(predictions)
print(labels)

# 打印损失
print("损失值：", loss.item())

aa = torch.tensor([[1,2,3],
                    [-1,2,3]])
mask = (aa>0)
mask_aa = aa[mask]

ab = torch.tensort(float('nan'))
if torch.isnan(ab):
    print('nan')