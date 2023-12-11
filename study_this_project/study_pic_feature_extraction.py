import torch
import torch.nn as nn

input_pic = torch.randn(2,20000,3)


class PicFeatureExtractorNet(nn.Module):
    def __init__(self,output_feature_dim) -> None:
        super().__init__()
        self.output_feature_dim = output_feature_dim #指定输出的维度，输入维度是batchsize*num_point*3
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, self.output_feature_dim)
        self.relu = nn.ReLU()
   
    def forward(self,pic):
        # pic = pic.view(pic.size(0),-1)
        pic = self.relu(self.fc1(pic))
        pic = self.relu(self.fc2(pic))
        pic = self.fc3(pic)

        pic_feature = pic.view(pic.size(0),-1,self.output_feature_dim)
        return pic_feature

pic_feature_extractor_net = PicFeatureExtractorNet(50)

pic_feature = pic_feature_extractor_net(input_pic)
print(pic_feature.shape)