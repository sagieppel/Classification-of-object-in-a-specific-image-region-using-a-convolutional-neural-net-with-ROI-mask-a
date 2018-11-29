# Trained net for region specific Classification
# 1. If not already set, download and set coco API and data set (See instruction)
# 1. Set Train image folder path  in: TrainImageDir
# 2. Set the path to the coco Train annotation json file in: TrainAnnotationFile
# 3. Run the script
# 4. The trained net weight will appear in the folder defined in: logs_dir
# 5. For other training parameters see Input section in train.py script
#------------------------------------------------------------------------------------------------------------------------
# Getting COCO dataset and API
# Download and extract the [COCO 2014 train images and Train/Val annotations](http://cocodataset.org/#download)
# Download and make the COCO python API base on the instructions in (https://github.com/cocodataset/cocoapi).
# Copy the pycocotools from cocodataset/cocoapi to the code folder (replace the existing pycocotools folder in the code).
# Note that the code folder already contain pycocotools folder with a compiled API that may or may not work as is.
##########################################################################################################################################################################
import numpy as np
import Resnet50Attention as Net
import COCOReader as COCOReader

import os
import scipy.misc as misc
import torch
#...........................................Input Parameters.................................................
UseCuda=True
TrainImageDir='/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/train2014/' #
TrainAnnotationFile = '/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/annotations/instances_train2014.json'
MinSize=160  # max width/hight of image
MaxSize=800 # min width/hight of image
MaxBatchSize=20  # Maximoum number of images per batch h*w*bsize (reduce to prevent oom problems)
MaxPixels=800*800*4 # Maximoum number of pixel per batch h*w*bsize (reduce to prevent oom problems)
logs_dir= "logs/"# "path to logs directory where trained model and information will be stored"
if not os.path.exists(logs_dir): os.makedirs(logs_dir)

Trained_model_path="" # If you want  to  training start from pretrained model other wise set to =""

Learning_Rate=1e-5 #Learning rate for Adam Optimizer
learning_rate_decay=0.999999#
#-----------------------------Other Paramters------------------------------------------------------------------------
TrainLossTxtFile=logs_dir+"TrainLoss.txt" #Where train losses will be writen
ValidLossTxtFile=logs_dir+"ValidationLoss.txt"# Where validation losses will be writen
Weight_Decay=1e-5# Weight for the weight decay loss function
MAX_ITERATION = int(1000010) # Max  number of training iteration

#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader=COCOReader.COCOReader(TrainImageDir,TrainAnnotationFile, MaxBatchSize,MinSize,MaxSize,MaxPixels)
NumClasses = Reader.NumCats

#---------------------Initiate neural net------------------------------------------------------------------------------------
Net=Net.Net(NumClasses=NumClasses,UseGPU=UseCuda)
Net.AddAttententionLayer()
if Trained_model_path!="":
    Net.load_state_dict(torch.load(Trained_model_path))
if UseCuda: Net.cuda()
#optimizer=torch.optim.SGD(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay,momentum=0.5)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay)
#--------------------------- Create files for saving loss----------------------------------------------------------------------------------------------------------
f = open(TrainLossTxtFile, "a")
f.write("Iteration\tloss\t Learning Rate="+str(Learning_Rate))
f.close()
AVGLoss=0
#..............Start Training loop: Main Training.............................................................................................
for itr in range(1,MAX_ITERATION):
    Images,SegmentMask,Labels, LabelsOneHot=Reader.ReadNextBatchRandom()
#**********************************************************************************************************************************
    # Images[:,:,:,1]*=SegmentMask
    # for ii in range(Labels.shape[0]):
    #     print(Reader.CatNames[Labels[ii]])
    #     misc.imshow(Images[ii])
#**************************Run Trainin cycle***************************************************************************************
    Prob, Lb=Net.forward(Images,ROI=SegmentMask) # Run net inference and get prediction
    Net.zero_grad()
    OneHotLabels=torch.autograd.Variable(torch.from_numpy(LabelsOneHot).cuda(), requires_grad=False)
    Loss = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))  # Calculate cross entropy loss
    if AVGLoss==0:  AVGLoss=float(np.array(Loss.data)) #Caclculate average loss for display
    else: AVGLoss=AVGLoss*0.999+0.001*float(np.array(Loss.data))
    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient decend change weight
    torch.cuda.empty_cache()
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 10000 == 0 and itr>0:
        print("Saving Model to file in "+logs_dir)
        torch.save(Net.state_dict(), logs_dir+ "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
    if itr % 10==0: # Display train loss
        print("Step "+str(itr)+" Train Loss="+str(float(np.array(Loss.data)))+" Runnig Average Loss="+str(AVGLoss))
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write("\n"+str(itr)+"\t"+str(float(np.array(Loss.data)))+"\t"+str(AVGLoss))
            f.close()
##################################################################################################################################################

