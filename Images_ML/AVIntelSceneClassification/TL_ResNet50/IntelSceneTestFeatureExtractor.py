
# Extract features of Test Intel Scene Images

from keras.applications import ResNet50
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

## get the test files.
from my_utils.datasets import DataSetLoader
import pandas as pd

IMAGE_DIR = 'dataset/train-scene classification'
TRAIN_IMAGE_LOC = IMAGE_DIR+'/train'

test_csv_df = pd.read_csv(IMAGE_DIR+'/test.csv')

DataLoader = DataSetLoader()
test_files = DataLoader.get_filename_list(TRAIN_IMAGE_LOC, list(test_csv_df['image_name']))

print(len(test_files))


## Get the test ids.

test_ids = list(test_csv_df['image_name'])
# remove .jpg extension.
test_ids = [w.replace('.jpg', '') for w in test_ids]

# convert the string to int
test_ids = [int(w) for w in test_ids]

## configurations

NUM_FILES = len(test_files) - 1

BATCH_SIZE = 25

## HDF5 writer
from my_utils.io import HDF5DataSetWriter
#resnet50 -> predict returns vector of size 2048=
db = HDF5DataSetWriter( (NUM_FILES, 2048), "./output/IntelSceneTestExtractedFeatures.hdf5",
                               dataKey="test_features",
                               valueKey='ID')

#### extract the features
import cv2
import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img

## set up the progress bar
import progressbar

# initialize the progress bar
widgets = ["Extracting Test Features: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]

pbar = progressbar.ProgressBar(maxval=NUM_FILES,
                                widgets=widgets).start()

# go over all images
for i in np.arange(0, NUM_FILES, BATCH_SIZE):

    # process them in terms of batches
    batchImages = []

    batchFileList = test_files[i: i+BATCH_SIZE]
    batchLabelList = test_ids[i: i+BATCH_SIZE]
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
    pbar.update(i)


db.close()
pbar.finish()
