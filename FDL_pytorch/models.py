import torch
import torch.nn as nn
from torchvision import models as tv
import torchvision


class VGG(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(VGG, self).__init__()
        
        if torchvision.__version__ >= "0.13":
            vgg_pretrained_features = tv.vgg19(weights='VGG19_BN_Weights.IMAGENET1K_V1').features
        else:
            vgg_pretrained_features = tv.vgg19(pretrained=pretrained).features
            
        # print(vgg_pretrained_features)
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        # vgg19
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])                
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [64,128,256,512,512]

  
    def get_features(self, x):
        # normalize the data
        h = (x-self.mean.to(x))/self.std.to(x)
        
        h = self.stage1(h)
        h_relu1_2 = h
        
        h = self.stage2(h)
        h_relu2_2 = h
        
        h = self.stage3(h)
        h_relu3_3 = h
        
        h = self.stage4(h)
        h_relu4_3 = h

        h = self.stage5(h)
        h_relu5_3 = h

        # get the features of each layer
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

        return outs
       
    def forward(self, x):
        feats_x = self.get_features(x)
        return feats_x


class ResNet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(ResNet, self).__init__()

        model = tv.resnet101(pretrained=pretrained)
        model.eval()
        # print(model)

        self.stage1 =nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu
        )
        self.stage2 = nn.Sequential(
            model.maxpool,
            model.layer1,
        )
        self.stage3 = nn.Sequential(
            model.layer2,
        )
        self.stage4 = nn.Sequential(
            model.layer3,
        )
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [64,256,512,1024]#

    def get_features(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]#
        return outs

       
    def forward(self, x):
        feats_x = self.get_features(x)
        return feats_x
    
class Inception(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(Inception, self).__init__()
        inception = tv.inception_v3(pretrained=pretrained, aux_logits=False)
            
        # print(inception)
        self.stage1 = torch.nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,      
        )
        self.stage2 = torch.nn.Sequential(
            inception.maxpool1,
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3, 
        )
        self.stage3 = torch.nn.Sequential(
            inception.maxpool2, 
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
        )
        self.stage4 = torch.nn.Sequential(
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        )
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
        self.chns = [64,192,288,768]
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

  
    def get_features(self, x):
        h = (x-self.mean)/self.std
        # h = (x-0.5)*2
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3,h_relu4_3]
        return outs
       
    def forward(self, x):
        feats_x = self.get_features(x)
        return feats_x
    
class EffNet(torch.nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()
        model = tv.efficientnet_b7(pretrained=True).features#[:6]
        model.eval()
        # print(model)
        self.stage1 = model[0:2]
        self.stage2 = model[2]
        self.stage3 = model[3]
        self.stage4 = model[4]
        self.stage5 = model[5]

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        for param in self.parameters():
            param.requires_grad = False
        self.chns = [32, 48, 80, 160, 224]
       
    def get_features(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h        
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]#           
        return outs
       
    def forward(self, x):
        feats_x = self.get_features(x)
        return feats_x