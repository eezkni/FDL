import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import VGG, ResNet, Inception, EffNet

class FDL_loss(torch.nn.Module):
    def __init__(self, patch_size=5, stride=1, num_proj=256, model = 'VGG', phase_weight = 1.0):
        '''
        patch_size, stride, num_proj: SWD slice parameters
        model: feature extractor, support VGG, ResNet, Inception, EffNet
        phase_weight: weight for phase branch
        '''
        
        super(FDL_loss, self).__init__()
        if model == 'ResNet':
            self.model = ResNet()
        elif model == 'EffNet':
            self.model = EffNet()
        elif model == 'Inception':
            self.model = Inception()
        elif model == 'VGG':
            self.model = VGG()
        else:
            assert "Invalid model type! Valid models: VGG, Inception, EffNet, ResNet"
            
        self.phase_weight = phase_weight
        self.stride = stride
        for i in range(len(self.model.chns)):
            rand = torch.randn(num_proj, self.model.chns[i], patch_size, patch_size)
            rand = rand/rand.view(rand.shape[0], -1).norm(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            self.register_buffer('rand_{}'.format(i), rand)
            
        # print all the parameters
        
    def forward_once(self, x, y, idx):
        rand= self.__getattr__('rand_{}'.format(idx))
        projx = F.conv2d(x, rand, stride=self.stride)
        projx = projx.reshape(projx.shape[0],projx.shape[1], -1)
        projy = F.conv2d(y, rand, stride=self.stride)
        projy = projy.reshape(projy.shape[0],projy.shape[1], -1)
        
        # sort the convolved input
        projx, _ = torch.sort(projx,dim=-1)
        projy, _ = torch.sort(projy,dim=-1)

        # compute the mean of the sorted convolved input
        s = torch.abs(projx - projy).mean([1,2])

        return s

    
    def forward(self, x, y):
        x = self.model(x)
        y = self.model(y)
        score = []
        for i in range(len(x)):
            # Transform to Fourier Space
            fft_x = torch.fft.fftn(x[i], dim=(-2, -1))
            fft_y = torch.fft.fftn(y[i], dim=(-2, -1))
            
            x_mag = torch.abs(fft_x)
            x_phase = torch.angle(fft_x)
            y_mag = torch.abs(fft_y)
            y_phase = torch.angle(fft_y)
            
            s_energy = self.forward_once(x_mag, y_mag, i)
            s_phase = self.forward_once(x_phase, y_phase, i)
            
            score.append(s_energy + s_phase*self.phase_weight)
            
        score = sum(score) # sumup between different layers
        score = score.mean() # mean within batch
        return score

# if __name__ == '__main__':
#     print("FDL_loss")
#     X = torch.randn((1, 3,128,128)).cuda()
#     Y = torch.randn((1, 3,128,128)).cuda() * 2
#     print("input_image shape: ",input_image.shape)

#     loss = FDL_loss().cuda()
#     c = loss(input_image, target_image)
#     print('loss:', c)
#     d = 0

    
