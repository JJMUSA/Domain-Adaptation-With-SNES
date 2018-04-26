import os
from PIL import Image
import numpy as np
import scipy.misc
#from utils import *
from  domain_adaptation import DomainAdaptation
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense

amazonPATH = "Original_images\\amazon"
dslrPATH = "Original_images\\dslr"
webcamPATH = "Original_images\\webcam"


def get_image_list(dic):
    print("Reading Data: Might take a while")
    image_list = []
    image_label = []
    for root, dirs, files in os.walk(dic):
        for file in files:
            if file.endswith('.jpg'):
                # print file
                temp = str(root).split('/')
                label = temp[-1]
                label = label.split('\\')
                label = label[-1]
                im = Image.open(root + '/' + file)
                im = np.array(im)
                im= scipy.misc.imresize(im, [84,84])
                image_list.append(im)
                image_label.append(label)
    return np.array(image_list), np.array(image_label)


amazonImage, amazonLabels = get_image_list(amazonPATH)
dslrImages, dslrLabels = get_image_list(dslrPATH)
webcamImages, webcamLabels = get_image_list(webcamPATH)

def get_label_indices(all_label):
    ''' returns the starting index of each label'''
    unique_labels, counts = np.unique(all_label, return_counts=True)
    indices = list(counts)
    indices[0]=0
    for i in range(1,len(counts)):
        indices[i]= indices[i-1] + counts[i-1]
    return indices

def sample_data(start_index, ending_index,training_size,testing_size):
    '''sample random indices for train, test and validation for each folder '''
    indices = list(range(start_index,ending_index ))
    training_size, testing_size = int(training_size*len(indices)), int(testing_size*len(indices))
    training_indices = np.random.choice(a=indices, size=training_size)
    remain_indices = [ i for i in indices  if i not in training_indices]
    testing_indices = np.random.choice(a=remain_indices,size=testing_size)
    valid_indices=np.asarray([i for i in remain_indices if i not in testing_indices])
    return (training_indices, valid_indices, testing_indices)

def split_data(label_indices,labels,train_size,test_size):
    '''create a train,test and validation set for all the categorises '''
    train_indices = []
    valid_indices= []
    testing_indices = []
    for i in range(0, len(label_indices)):
        ind = sample_data(label_indices[i], label_indices[i + 1], train_size, test_size) if i < len(dslr_label_indices) - 1 else sample_data(dslr_label_indices[i], len(labels), train_size, test_size)
        train_indices.append(ind[0])
        valid_indices.append(ind[1])
        testing_indices.append(ind[2])
    train_indices = np.concatenate(train_indices,axis=0)
    valid_indices = np.concatenate(valid_indices, axis=0)
    testing_indices = np.concatenate(testing_indices, axis=0)

    np.random.shuffle(train_indices)
    np.random.shuffle(valid_indices)
    np.random.shuffle(testing_indices)
    return(train_indices, valid_indices, testing_indices)

dslr_label_indices =  get_label_indices(dslrLabels)
amazon_label_indices = get_label_indices(amazonLabels)
webcam_label_indices = get_label_indices(webcamLabels)



dslrTraining_indices, dslrValid_indices, dslrTest_indices =split_data(dslr_label_indices,dslrLabels,0.3,0.6)
amazonTraining_indices, amazonValid_indices, amazonTest_indices =  split_data(amazon_label_indices,amazonLabels,0.6,0.35)
webcamTraining_indices, webValid_indices, amazonTest_indices = split_data(webcam_label_indices,webcamLabels, 0.5,0.4)


amazonTrainX, amazonTrainY = amazonImage[amazonTraining_indices], amazonLabels[amazonTraining_indices]
amazonValidX, amazonValidY =  amazonImage[amazonTraining_indices], amazonLabels[amazonTraining_indices]
amazonTestX, amazonTestY = amazonImage[amazonTraining_indices], amazonLabels[amazonTraining_indices]

max_amazon_pixel=np.amax(amazonImage)
amazonTrainX = amazonTrainX.astype('float32') *(255 / max_amazon_pixel)
amazonValidX =  amazonValidX.astype('float32') * (255 / max_amazon_pixel)
amazonTestX = amazonTestX.astype('float32') * (255 / max_amazon_pixel)



dslrTrainX, dslrTrainY = dslrImages[dslrTraining_indices], dslrLabels[dslrTraining_indices]
dslrValidX, dslrValidY = dslrImages[dslrValid_indices], dslrLabels[dslrValid_indices]
dslrTestX, dslrTestY = dslrImages[dslrTest_indices], dslrLabels[dslrTest_indices]

max_dslr_pixel = np.amax(dslrImages)
dslrTrainX = dslrTrainX.astype('float32') * (255 / max_dslr_pixel)
dslrValidX = dslrValidX.astype('float32') * (255 / max_dslr_pixel)
dslrTestX = dslrTestX.astype('float32') * (255 / max_dslr_pixel)


print("--Training with amazon and tesing on dslr--")
print("using %d instances of amazon data for training and testing on %d instance of dslr data" %(amazonTrainX.shape[0], dslrTestX.shape[0]))
source = [(amazonTrainX, amazonTrainY), (amazonValidX, amazonValidY)]
target = [(dslrTrainX, dslrTrainY), (dslrValidX, dslrValidY)]

input_shape = amazonImage.shape[1:]
feature_extractor = Sequential()
feature_extractor.add(Conv2D(16, kernel_size=(8,8), activation= 'relu',input_shape= input_shape, strides=(4,4)))
feature_extractor.add(Conv2D(32,kernel_size=(4,4), activation='relu',strides=(2,2)))
feature_extractor.add(Flatten())
feature_extractor.add(Dense(2,activation='relu'))



amazon_dslr_adaptation = DomainAdaptation(source, target,feature_extractor)
amazon_dslr_adaptation.train()
# testing
print("accuarcy     ")

