import os 
from PIL import Image
import numpy as np
from utils import *
amazonPATH="C:\\Users\\cc\\Anaconda3\\envs\\MLenviroment\\CE888\Original_images\\amazon"
dslrPATH="C:\\Users\\cc\\Anaconda3\\envs\\MLenviroment\\CE888\Original_images\\dslr"
webcamPATH="C:\\Users\\cc\\Anaconda3\\envs\\MLenviroment\\CE888\Original_images\\webcam"
def get_image_list(dic):
    image_list = []
    image_label = []
    for root, dirs, files in os.walk(dic):
        for file in files:
            if file.endswith('.jpg'):
                # print file
                temp = str(root).split('/')
                label = temp[-1]
                im = Image.open(root + '/' + file)
                im=np.array(im)
				
                image_list.append(im)
                image_label.append(label)
    return np.array(image_list), np.array(image_label)
	
	
	
amazonImage,amazonLabels=get_image_list(amazonPATH)
dslrImages,dslrLabels=get_image_list(dslrPATH)
webcamImages,webcamLabels=get_image_list(webcamPATH)
print(amazonImage[0].shape)
print(dslrImages[0].shape)
print(webcamImages[0].shape)
#display=[dslrImages[0],dslrImages[30],dslrImages[54],dslrImages[60]]
#im=imshow_grid(display)
#im.save("amazonData.jpeg")
#im.show()
