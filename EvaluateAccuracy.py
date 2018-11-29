#Evaluate top one accuracy prediction prediction of the net using the coco evaluation data set

# 1. Train net or Download pre trained net weight from [here](https://drive.google.com/file/d/1xRFvBk_PONwJHmP2NcwFaEQc_Z_JInpE/view?usp=sharing).
# 2. If not already set, download and set coco API and data set (See instruction)
# 3. Set Train coco image folder path  in: TestImageDir
# 4. Set the path to the coco Train annotation json file in: TestAnnotationFile
# 5. Run the script

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
import numpy as np

#...........................................Input Parameters.................................................

Trained_model_path="logs/WeightsRegionSpecificClassification.torch" # Weights for Pretrained net
TestImageDir='/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/val2014/' #COCO evaluation image dir
TestAnnotationFile = '/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/annotations/instances_val2014.json' # COCO Val annotation file
EvaluationFile="PrecisionStatistics.xls"
SamplePerClass=2
UseCuda=True

#---------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader=COCOReader.COCOReader(TestImageDir,TestAnnotationFile)
NumClasses = Reader.NumCats

#---------------------Initiate neural net------------------------------------------------------------------------------------
Net=Net.Net(NumClasses=NumClasses,UseGPU=UseCuda)
Net.AddAttententionLayer() #Load attention layers

Net.load_state_dict(torch.load(Trained_model_path)) #load weights
if UseCuda: Net.cuda()
Net.eval()
#--------------------Evaluate net accuracy---------------------------------------------------------------------------------
#===============================Sizes bins============================================================================================
Sizes=[1000,2000,4000,8000,16000,32000,64000,128000,256000,500000,1000000] #sizes pixels
NumSizes=len(Sizes)
#--------------------Evaluate net accuracy---------------------------------------------------------------------------------
TP=np.zeros([Reader.NumCats+1],dtype=np.float64) # True positive per class
FP=np.zeros([Reader.NumCats+1],dtype=np.float64) # False positive per class
FN=np.zeros([Reader.NumCats+1],dtype=np.float64) # False Negative per class
SumPred=np.zeros([Reader.NumCats+1],dtype=np.float64) #SumCases Per class

SzTP=np.zeros([Reader.NumCats+1,NumSizes],dtype=np.float64) # True positive per class per size
SzFP=np.zeros([Reader.NumCats+1,NumSizes],dtype=np.float64) # False positive per class per size
SzFN=np.zeros([Reader.NumCats+1,NumSizes],dtype=np.float64) # False Negative per class per size
SzSumPred=np.zeros([Reader.NumCats+1,NumSizes],dtype=np.float64)
#--------------------Evaluate net accuracy---------------------------------------------------------------------------------
CorCatPred=np.zeros([Reader.NumCats],dtype=np.float64) # Counter of currect class prediction
TotalCat=np.zeros([Reader.NumCats],dtype=np.float64)
for c in range(Reader.NumCats):
      print("Class "+str(c)+") "+Reader.CatNames[c]+"Num Casses "+str(np.min((SamplePerClass,len(Reader.ImgIds[c])))))
      for m in range(np.min((SamplePerClass,len(Reader.ImgIds[c])))):
            Images,SegmentMask,Labels, LabelsOneHot=Reader.ReadSingleImageAndClass(ClassNum=c,ImgNum=m)
            print("Class " + str(c) + ") " + Reader.CatNames[c]+"  "+str(m))
            BatchSize = Images.shape[0]
            for i in range(BatchSize):
              #    print(i)
                  # ..................................................................
                  Prob, Lb = Net.forward(Images[i:i + 1], ROI=SegmentMask[i:i + 1],
                                         EvalMode=True)  # Run net inference and get prediction
                  PredLb = np.array(Lb.data)
                  # .......................................................................................
                  LbSize = SegmentMask[i].sum()
                  SzInd = -1
                  for f, sz in enumerate(Sizes):
                        if LbSize < sz:
                              SzInd = f
                              break

                  if PredLb[0] == Labels[i]:
                        #   print("Correct")
                        TP[Labels[i]] += 1
                        SzTP[Labels[i], SzInd] += 1
                  else:
                        #     print("Wrong")
                        FN[Labels[i]] += 1
                        FP[PredLb[0]] += 1
                        SzFN[Labels[i], SzInd] += 1
                        SzFP[PredLb[0], SzInd] += 1
                  SumPred[Labels[i]] += 1
                  SzSumPred[Labels[i], SzInd] += 1
                  # Images[i,:,:,0]*=1-SegmentMask[i]
                  # print("Real Label " + str(Reader.MaterialDict[Labels[i]]) + "   Predicted " + str(Reader.MaterialDict[PredLb[0]]))
                  # misc.imshow(Images[i])
                  #  print("I love beer mister hazir")

# =====================================================================================================
f = open(EvaluationFile, "w")

NrmF = len(SumPred) / (np.sum(SumPred > 0))  # Normalization factor for classes with zero occurrences

txt = "Mean Accuracy All Class Average =\t" + str(
      (TP / (SumPred + 0.00000001)).mean() * NrmF * 100) + "%" + "\r\n"
print(txt)
f.write(txt)

txt = "Mean Accuracy Images =\t" + str((TP.mean() / SumPred.mean()) * 100) + "%" + "\r\n"
print(txt)
f.write(txt)

print("\r\n=============================================================================\r\n")
print(txt)
f.write(txt)

txt = "SizeMax\tMeanClasses\tMeanGlobal\tNum Instances\tNumValidClasses\r\n"
print(txt)
f.write(txt)
for i, sz in enumerate(Sizes):
      if SzSumPred[:, i].sum() == 0: continue
      NumValidClass = np.sum(SzSumPred[:, i] > 0)
      NrmF = len(SzSumPred[:, i]) / NumValidClass  # Normalization factor for classes with zero occurrences
      txt = str(sz) + "\t" + str((SzTP[:, i] / (SzSumPred[:, i] + 0.00001)).mean() * NrmF * 100) + "%\t" + str(
            100 * (SzTP[:, i]).mean() / (SzSumPred[:, i].mean())) + "%\t" + str(
            SzSumPred[:, i].sum()) + "\t" + str(NumValidClass) + "\r\n"
      print(txt)
      f.write(txt)
f.close()
