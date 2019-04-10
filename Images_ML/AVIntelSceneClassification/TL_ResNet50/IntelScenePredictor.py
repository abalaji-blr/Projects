
## Intel Scene Predictor

## Load the model file
import pickle
model_file = './model/IntelSceneClassifier.pickle'
f = open(model_file, 'rb')
model = pickle.load(f)

print(model)

## Read Test Extracted Features
import h5py

test_feature_db = './output/IntelSceneTestExtractedFeatures.hdf5'

#open the hdf5 file.
db = h5py.File(test_feature_db, 'r')

## Do the decode_predictions
print('Predict the test images\n')

preds = model.predict(db['test_features'])

### submission file
print("Generate submission file...")

file = open("./submission.csv", "w")

file.write("image_name,label\n")

for i in range(0, len(db['ID'])):
    #get the ID
    id = db['ID'][i]
    img_name = str(id) + '.jpg'

    # write the predictions
    file.write('%s,' % img_name)
    file.write('%d\n' % preds[i]);


## the last one image.
file.write('24333.jpg,0\n')
#close the streams.
file.close()

## close all the streams
db.close();
f.close()

print('Done!')
