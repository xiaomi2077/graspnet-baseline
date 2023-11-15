import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成预测值和实际值的一系列角度
pred_values = torch.linspace(0, 180, 180)  # 预测值范围
gt_values = torch.linspace(0, 180, 180)    # 实际值范围

# 计算损失
loss_values = torch.zeros((180, 180))
for i, pred in enumerate(pred_values):
    for j, gt in enumerate(gt_values):
        loss_values[i, j] = min(abs(pred - gt), abs(180 - abs(pred - gt)))

# 创建网格
pred_mesh, gt_mesh = torch.meshgrid(pred_values, gt_values)

# 绘制三维图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(pred_mesh.numpy(), gt_mesh.numpy(), loss_values.numpy(), cmap='viridis')

# 设置坐标轴标签
ax.set_xlabel('pred_val')
ax.set_ylabel('gt_val')
ax.set_zlabel('loss')
ax.set_title(r'$\min(|\text{pred} - \text{gt}|, |180 - |\text{pred} - \text{gt}||)$ img')

plt.show()