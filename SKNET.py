import torch
import torch.nn as nn
from torchsummary import summary

class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 , stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(#nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                #nn.Linear(features, d),
                                nn.Conv1d(features, d, kernel_size=1, stride=1),
                                nn.BatchNorm1d(d),
                                nn.ReLU(inplace=False),
                                nn.Conv1d(d, 2 * features, kernel_size=1, stride=1))
        # self.fcs = nn.ModuleList([])
        # for i in range(M):
        #     self.fcs.append(
        #           nn.Conv1d(d, features, kernel_size=1, stride=1)
        #           #nn.Linear(d, features)
        #     )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size,in_ch,W,H = x.shape
        # Split
        feats_3x3 = self.convs[0](x)
        feats_5x5 = self.convs[1](x)
        # feats = [conv(x) for conv in self.convs]
        # feats = torch.cat(feats, dim=1)
        # feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        
        # Fuse
        feats_U = feats_5x5.add(feats_3x3)
        feats_S = self.gap(feats_U).view([batch_size, self.features])
        #print(feats_S.shape)
        feats_Z = self.fc(feats_S.unsqueeze(2))
        
        # Select
        # attention_vectors = [fc(feats_Z) for fc in self.fcs]
        # attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = feats_Z
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        # print(attention_vectors.shape)
        attention_vectors = self.softmax(attention_vectors)
        att_3x3 = attention_vectors[:,0]
        att_5x5 = attention_vectors[:,1]
        # feats_V = torch.sum(feats*attention_vectors, dim=1)
        feats_3x3 = feats_3x3.mul(att_3x3)
        feats_5x5 = feats_5x5.mul(att_5x5)
        feats_V = feats_3x3.add_(feats_5x5)
        
        return feats_V


class SKUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            )
        
        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=stride, L=L)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
            )
        

        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)
        
        return self.relu(out + self.shortcut(residual))

class SKNet(nn.Module):
    def __init__(self, class_num, nums_block_list = [3, 4, 6,3], strides_list = [1, 2, 2, 2], G = 32):
        '''
        Parameters
        ----------
        class_num : INT, output layer size(number of classes)
        nums_block_list : List, number of SKUnit in each block(max size of list = 4), default is [3, 4, 6, 3].
        strides_list : List, number of strides for SKUnit in each block, default is [1, 2, 2, 2].

        '''
        super(SKNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding = 1, bias=False, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
                
        self.stage_1 = self._make_layer(64, 64, 64, nums_block=nums_block_list[0], stride=strides_list[0], G = G)
        self.stage_2 = self._make_layer(64, 128, 128, nums_block=nums_block_list[1], stride=strides_list[1], G = G)
        self.stage_3 = self._make_layer(128, 256, 256, nums_block=nums_block_list[2], stride=strides_list[2], G = G)
        self.stage_4 = self._make_layer(256, 512, 512, nums_block=nums_block_list[3], stride=strides_list[3], G = G)
     
        self.gap = nn.AvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512, class_num))#,
            # nn.Softmax(dim = 1))
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    def _make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1, G = 32):
        layers=[SKUnit(in_feats, mid_feats, out_feats, stride=stride, G = G)]
        for _ in range(1,nums_block):
            layers.append(SKUnit(out_feats, mid_feats, out_feats, G = G))
        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        # print(fea.shape)
        fea = self.stage_4(fea)
        fea = self.gap(fea)
        # return fea
        fea = fea.view(fea.shape[0], -1)
        fea = self.classifier(fea)
        return fea
    
    
if __name__ == '__main__':
    net = SKNet(200, [2,2,2,2], [1,2,2,2], G = 1).cuda()
    # print(summary(net, (3, 64, 64)))
    print(summary(net, (3, 56, 56)))
    # c = SKConv(128)
    # x = torch.zeros(8,128,2,2)
    # print(c(x).shape)   