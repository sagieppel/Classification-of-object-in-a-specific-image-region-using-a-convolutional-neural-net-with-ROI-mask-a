#
#  Read image segment region and classes from the COCO data set  (need the coco API to run)

# Getting COCO dataset and API
# Download and extract the [COCO 2014 train images and Train/Val annotations](http://cocodataset.org/#download)
# Download and make the COCO python API base on the instructions in (https://github.com/cocodataset/cocoapi).
# Copy the pycocotools from cocodataset/cocoapi to the code folder (replace the existing pycocotools folder in the code).
# Note that the code folder already contain pycocotools folder with a compiled API that may or may not work as is.
#
#
import numpy as np
import os
import scipy.misc as misc
import random
from pycocotools.coco import COCO
import cv2
#------------------------Class for reading training and  validation data---------------------------------------------------------------------
class COCOReader:
################################Initiate folders were files are and list of train images############################################################################
    def __init__(self, ImageDir='/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/train2014/',AnnotationFile = '/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/COCO/annotations/instances_train2014.json', MaxBatchSize=100,MinSize=160,MaxSize=800,MaxPixels=800*800*5):
        self.ImageDir=ImageDir # Image dir
        self.AnnotationFile=AnnotationFile # File containing image annotation
        self.MaxBatchSize=MaxBatchSize # Max number of image in batch
        self.MinSize=MinSize # Min image width and hight
        self.MaxSize=MaxSize #MAx image width and hight
        self.MaxPixels=MaxPixels # Max number of pixel in all the batch (reduce to solve out of memory issues)
#-------------------initialize COCO api for instance annotations---------------------------------------------------------------------------
        self.dataType = 'train2014'
        self.coco = COCO(AnnotationFile ) #Load annotation file
        self.cats = self.coco.loadCats(self.coco.getCatIds())  # List of categories
        self.NumCats = len(self.cats)  # Num categories
        self.CatNames = [cat['name'] for cat in self.cats] # Categories names
        self.ImgIds={} # List of ids of images containing various of categories
        for i in range(self.NumCats): self.ImgIds[i] = self.coco.getImgIds(catIds=self.cats[i]['id']) # Create image list of each category
        self.ClassItr=np.zeros(self.NumCats)




##############################################################################################################################################
######################################Read next batch of images and labels with no augmentation but with croping and resizing pick random images labels and cropping (for training)######################################################################################################
    def ReadNextBatchRandom(self):
#=====================Set batch size=============================================================================================
        Hb=np.random.randint(low=self.MinSize,high=self.MaxSize) # Batch hight
        Wb=np.random.randint(low=self.MinSize,high=self.MaxSize) # batch  width
        BatchSize=np.int(np.min((np.floor(self.MaxPixels/(Hb*Wb)),self.MaxBatchSize))) # Number of images in batch
        BImgs=np.zeros((BatchSize,Hb,Wb,3)) # Images
        BSegmentMask=np.zeros((BatchSize,Hb,Wb)) # Segment mask
        BLabels=np.zeros((BatchSize),dtype=np.int) # Class
        BLabelsOneHot=np.zeros((BatchSize,self.NumCats),dtype=np.float32) # classes in one hot encodig
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#======================================================================================================================================
        for i in range(BatchSize):
#=========================Load image and annotation========================================================================================
            ClassNum = np.random.randint(self.NumCats)  # ] ['name']  # Chose random category
            ImgNum = np.random.randint(len(self.ImgIds[ClassNum]))  # Choose Random image
            ImgData = self.coco.loadImgs(self.ImgIds[ClassNum][ImgNum])[0]  # Pick image data
            image_name = ImgData['file_name']  # Get image name
            Img = cv2.imread(self.ImageDir + "/" + image_name)  # Load Image
            if (Img.ndim==2): #If grayscale turn to rgb
                  Img=np.expand_dims(Img,3)
                  Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3] # Get first 3 channels incase there are more
            annIds = self.coco.getAnnIds(imgIds=ImgData['id'], catIds= self.cats[ClassNum]['id'],iscrowd=None)  # Get list of annotation ids for image  (of the specific class)
            # annIds = coco.getAnnIds(imgIds=ImgData['id'], iscrowd=None)# Get list of annotation for image  (of the specific class)
            InsAnn = self.coco.loadAnns(annIds)  # array of instance annotation
            Ins = InsAnn[np.random.randint(len(InsAnn))]  # Choose Random instance
            Mask = self.coco.annToMask(Ins)  # Get mask (binary map)
            bbox = np.array(Ins['bbox']).astype(np.float32)  # Get Instanc Bounding box
#========================resize image if it two small to the batch size==================================================================================
            [h,w,d]= Img.shape
            Rs=np.max((Hb/h,Wb/w)) # Resize factor
            if Rs>1:# Resize image and mask
                h=int(np.max((h*Rs,Hb)))
                w=int(np.max((w*Rs,Wb)))
                Img=cv2.resize(Img,dsize=(w,h),interpolation = cv2.INTER_LINEAR)
                Mask=cv2.resize(Mask,dsize=(w,h),interpolation = cv2.INTER_NEAREST)
                bbox*=Rs.astype(np.float32)
#=======================Crop image to fit batch size===================================================================================
            x1 = int(np.floor(bbox[0]))
            Wbox = int(np.floor(bbox[2])) #Bounding box width
            y1 = int(np.floor(bbox[1]))
            Hbox = int(np.floor(bbox[3])) # Bounding box height
            if Wb>Wbox:
                Xmax=np.min((w-Wb,x1))
                Xmin=np.max((0,x1-(Wb-Wbox)))
            else:
                Xmin=x1
                Xmax=np.min((w-Wb, x1+(Wbox-Wb)))

            if Hb>Hbox:
                Ymax=np.min((h-Hb,y1))
                Ymin=np.max((0,y1-(Hb-Hbox)))
            else:
                Ymin=y1
                Ymax=np.min((h-Hb, y1+(Hbox-Hb)))
            if Xmax<Xmin:
                print("waaa")
            if Ymax < Ymin:
                print("dddd")


            x0=np.random.randint(low=Xmin,high=Xmax+1)
            y0=np.random.randint(low=Ymin,high=Ymax+1)
            # Img[:,:,1]*=Mask
            # misc.imshow(Img)
            Img=Img[y0:y0+Hb,x0:x0+Wb,:]
            Mask=Mask[y0:y0+Hb,x0:x0+Wb]
           # misc.imshow(Img)

#======================Random mirror flip===========================================================================================
            if random.random() < 0.0:  # Agument the image by mirror image
                   Img = np.fliplr(Img)
                   Mask = np.fliplr(Mask)
#=====================Add to Batch================================================================================================
            BImgs[i] = Img
            BSegmentMask[i,:,:] = Mask
            BLabels[i] = int(ClassNum)
            BLabelsOneHot[i,ClassNum] = 1
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#======================================================================================================================================
        return BImgs,BSegmentMask,BLabels, BLabelsOneHot




##############################################################################################################################################
######################################Read next batch. given an image number and a class the batch conssit on all the instance of the input class in the input image######################################################################################################
    def ReadSingleImageAndClass(self,ClassNum,ImgNum):
            ImgData = self.coco.loadImgs(self.ImgIds[ClassNum][ImgNum])[0]  # Pick image data
            image_name = ImgData['file_name']  # Get image name
            Img = misc.imread(self.ImageDir + "/" + image_name)  # Load Image
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels
            annIds = self.coco.getAnnIds(imgIds=ImgData['id'],catIds=self.cats[ClassNum]['id'],iscrowd=None) # Get list of annotation for image  (of the specific class)
            InsAnn = self.coco.loadAnns(annIds)  # create array of instance annotation
#==================Create Batch=================================================================================================
            [Hb, Wb, d] = Img.shape
            BatchSize=len(InsAnn)
            BImgs = np.zeros((BatchSize, Hb, Wb, 3))  # Images
            BSegmentMask = np.zeros((BatchSize, Hb, Wb))  # Segment mask
            BLabels = np.zeros((BatchSize),dtype=np.int)  # Class of batch
            BLabelsOneHot = np.zeros((BatchSize, self.NumCats),dtype=np.float32)  # Batch classes in one hot encodig
#=================Fill batch====================================================================================================
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            for i in range(BatchSize):
                BImgs[i]=Img
                BSegmentMask[i,:,:] = self.coco.annToMask(InsAnn[i])  # Get mask (binary map)
                BLabels[i] = ClassNum
                BLabelsOneHot[i, ClassNum] = 1 # Set one hot encoding label
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#======================================================================================================================================
            return BImgs,BSegmentMask,BLabels, BLabelsOneHot
