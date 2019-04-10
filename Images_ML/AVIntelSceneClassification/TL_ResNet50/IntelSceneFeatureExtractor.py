# feature extractor for Intel scene classification

# get training files
from my_utils.datasets import DataSetLoader
import pandas as pd

IMAGE_DIR = 'dataset/train-scene classification'
TRAIN_IMAGE_LOC = IMAGE_DIR+'/train'

train_csv_df = pd.read_csv(IMAGE_DIR+'/train.csv')

DataLoader = DataSetLoader()
training_files = DataLoader.get_filename_list(TRAIN_IMAGE_LOC, list(train_csv_df['image_name']))

### labels
train_labels = train_csv_df['label']

labelNames = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

##### configurations
NUM_FILES = len(training_files)
#NUM_FILES = 68
BATCH_SIZE = 34

## load resnet50
from keras.applications import ResNet50
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

## create the HDF5 database
from my_utils.io import HDF5DataSetWriter
#resnet50 -> predict returns vector of size 2048
db = HDF5DataSetWriter( (NUM_FILES, 2048), "./output/IntelSceneExtractedFeatures.hdf5", dataKey="features")

############### extract features ##########
import cv2
import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img

# go over all images
for i in np.arange(0, NUM_FILES, BATCH_SIZE):

    # process them in terms of batches
    batchImages = []

    batchFileList = training_files[i: i+BATCH_SIZE]
    batchLabelList = train_labels[i: i+BATCH_SIZE]
    #print('outer for loop : {} '.format(i))

    for fileName in batchFileList:
        # load the file
        img = cv2.imread(fileName)
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)

        # convert to array
        img = img_to_array(img)

        #before preprocess, expand the dim.
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)  # for resnet50

        batchImages.append(img)

    #process the batch images
    batchImages = np.vstack(batchImages)

    # extract the features for the batch
    extracted_features = model.predict(batchImages, batch_size=BATCH_SIZE)

    ## separate out the features for each image.
    ## resnet50 spits out vector of 2048 for each image
    extracted_features = extracted_features.reshape((extracted_features.shape[0], 2048) )

    # add to db
    db.add(extracted_features, batchLabelList)

db.close()

print('Done!')
