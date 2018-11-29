# Run Predicition on single image and ROI mask Using trained net display predicted class on screen

# 1. Train net or Download pre trained net weight from [here](https://drive.google.com/file/d/1xRFvBk_PONwJHmP2NcwFaEQc_Z_JInpE/view?usp=sharing).
# 2. Open EvaluateAccuracy.py
# 3. Set Train coco image folder path  in: TestImageDir
# 4. Set the path to the coco Train annotation json file in: TestAnnotationFile
# 5. Run the script
##########################################################################################################################################################################

import numpy as np
import Resnet50Attention as Net
import os
import scipy.misc as misc
import torch
import GetCOCOCatNames
import numpy as np

#...........................................Input Parameters.................................................

Trained_model_path="logs/WeightsRegionSpecificClassification.torch" # Pretrained model
ImageFile='TestImages/Test4/Image.png' #Input image
ROIMaskFile= 'TestImages/Test4/InputMask4.png' # Input ROI mas
UseCuda=True
#---------------------Get list of coco classes-----------------------------------------------------------------------------
CatNames=GetCOCOCatNames.GetCOCOCatNames()
#---------------------Initiate neural net------------------------------------------------------------------------------------
Net=Net.Net(NumClasses=CatNames.__len__(),UseGPU=UseCuda)
Net.AddAttententionLayer()

Net.load_state_dict(torch.load(Trained_model_path)) #Load net
if UseCuda: Net.cuda()
Net.eval()
#--------------------Read Image and segment mask---------------------------------------------------------------------------------
Images=misc.imread(ImageFile)
ROIMask=misc.imread(ROIMaskFile)

misc.imshow(Images) # Display image
misc.imshow(ROIMask*255) # Disply ROI mask

Images=np.expand_dims(Images,axis=0)
ROIMask=np.expand_dims(ROIMask,axis=0)
#-------------------Run Prediction----------------------------------------------------------------------------
Prob, PredLb = Net.forward(Images, ROI=ROIMask,EvalMode=True)  # Run net inference and get prediction
PredLb = np.array(PredLb.data)
Prob = np.array(Prob.data)
#---------------Print Prediction on screen--------------------------------------------------------------------------
print("Predicted Label " + CatNames[PredLb[0]])
print("Predicted Label Prob="+str(Prob[0,PredLb[0]]*100)+"%")



