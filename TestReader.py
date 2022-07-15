
import cv2
import numpy as np
import Reader
import cv2
import os
import OpenSurfacesClasses

clas_dic=OpenSurfacesClasses.CreateMaterialDict()
ImageDir="/media/breakeroftime/2T/Data_zoo/OpenSurface/OpenSurfaceMaterialsSmall/Images"
AnnotationDir="/media/breakeroftime/2T/Data_zoo/OpenSurface/OpenSurfaceMaterialsSmall/TestLabels/"
Reader = Reader.Reader(ImageDir=ImageDir, AnnotationDir=AnnotationDir)
NumClasses = Reader.NumClass
for t in range(100):
    Imgs, SegmentMask, Labels, LabelsOneHot = Reader.ReadNextBatchRandom(EqualClassProbabilty=True)
    #Imgs, SegmentMask, Labels, LabelsOneHot = Reader.ReadNextImageClean()
    for f in range(Imgs.shape[0]):
        print(f)
        if not os.path.exists(str(f)): os.mkdir(str(f))
       # print(Reader.cats[Labels[f]]['name'])
       # print(Reader.MaterialDict[Labels[f]])

        Imgs[f] = Imgs[f][..., :: -1]
        Img=cv2.resize(Imgs[f],[int(Imgs[f].shape[1]/2),int(Imgs[f].shape[0]/2)])
        ROI=cv2.resize(SegmentMask[f], [int(Imgs[f].shape[1] / 2), int(Imgs[f].shape[0] / 2)])


        # misc.imsave("InputMask"+str(f+1)+".png",SegmentMask[f].astype(np.uint8))
        Imgs[f,:,:,1]  *=1-SegmentMask[f]
        Imgs[f, :, :, 2] *= 1 - SegmentMask[f]
        cv2.imshow(clas_dic[Labels[f]],Imgs[f].astype(np.uint8))
        cv2.waitKey()
        cv2.destroyAllWindows()
        ####cv2.imwrite(str(t) + "/OverLay" + str(f) + ".png", Imgs[f])



