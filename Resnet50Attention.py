#
#  Build resnet 50 neural net with attention mask directed classification (Classiffy only the image region marked in the mask)
#  
#
import scipy.misc as misc
import torchvision.models as models
import torch
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):# Net for region based segment classification
######################Load main net (resnet 50) class############################################################################################################
        def __init__(self,NumClasses, UseGPU=True): # Load pretrained encoder and prepare net layers
            super(Net, self).__init__()
            self.UseGPU = UseGPU # Use GPu with cuda
# ---------------Load pretrained torchvision resnet (need to be connected to the internet to download model in the first time)----------------------------------------------------------
            self.Net = models.resnet50(pretrained=True)
#----------------Change Final prediction fully connected layer from imagnet 1000 classes to coco 80 classes------------------------------------------------------------------------------------------
            self.Net.fc=nn.Linear(2048, NumClasses)

            self.Attentionlayer = nn.Conv2d(1, 64, stride=1, kernel_size=3, padding=1, bias=True)  # Generate attention filters their output will be multiplied with the main net feature map
            self.Attentionlayer.bias.data = torch.ones(self.Attentionlayer.bias.data.shape)
            self.Attentionlayer.weight.data = torch.zeros(self.Attentionlayer.weight.data.shape)

###############################################Run prediction inference using the net ###########################################################################################################
        def forward(self,Images,ROI):

#------------------------------- Convert from numpy to pytorch-------------------------------------------------------
                InpImages = torch.autograd.Variable(torch.from_numpy(Images), requires_grad=False).transpose(2,3).transpose(1, 2).type(torch.FloatTensor)
                ROImap = torch.autograd.Variable(torch.from_numpy(ROI.astype(np.float)), requires_grad=False).unsqueeze(dim=1).type(torch.FloatTensor)
                if self.UseGPU == True: # Convert to GPU
                    InpImages = InpImages.cuda()
                    ROImap = ROImap.cuda()
# -------------------------Normalize image-------------------------------------------------------------------------------------------------------
                RGBMean = [123.68, 116.779, 103.939]
                RGBStd = [65, 65, 65]
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # Normalize image by std and mean



#============================Run net layers===================================================================================================

                x=InpImages
#
                x = self.Net.conv1(x) # First resnet convulotion layer
                 #----------------Apply Attention layers--------------------------------------------------------------------------------------------------
                AttentionMap = self.Attentionlayer(F.interpolate(ROImap, size=x.shape[2:4], mode='bilinear'))
                x = x + AttentionMap

                # ---------------------First resnet block-----------------------------------------------------------------------------------------------
                x = self.Net.bn1(x)
                x = self.Net.relu(x)
                x = self.Net.maxpool(x)
                x = self.Net.layer1(x)
                # --------------------Second Resnet 50 Block------------------------------------------------------------------------------------------------
                x = self.Net.layer2(x)
                x = self.Net.layer3(x)
                # -----------------Resnet 50 block 4---------------------------------------------------------------------------------------------------
                x = self.Net.layer4(x)
                # ----------------Apply Attention layer--------------------------------------------------------------------------------------------------
                # ------------Fully connected final vector--------------------------------------------------------------------------------------------------------
                x = torch.mean(torch.mean(x, dim=2), dim=2)
                #x = x.squeeze()
                x = self.Net.fc(x)
                #---------------------------------------------------------------------------------------------------------------------------
                ProbVec = F.softmax(x,dim=1) # Probability vector for all classes
                Prob,Pred=ProbVec.max(dim=1) # Top predicted class and probability


                return ProbVec,Pred
###################################################################################################################################


