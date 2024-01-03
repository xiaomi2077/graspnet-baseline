import torch
import torch.nn.functional as F
from utils.smooth_label import angle_smooth_label_multi

target_index = torch.randint(0,12,size=(2,1024,4)) #shape 2*1024*4
target_smooth = torch.from_numpy(angle_smooth_label_multi(target_index.to(torch.float), 12,0,1,1)).permute(0,2,1,3) 

pred = torch.rand(2,12,1024,4,dtype=torch.float32) #shape 2*12*1024*4

loss_index = F.cross_entropy(pred,target_index)
loss_smooth = F.cross_entropy(pred,target_smooth,reduction='none')

print(loss_index)
print(loss_smooth)