# Load the model
import pickle

model_file = './model/cats_and_dogs.pickle'


## open the model file and load it.
f = open(model_file, 'rb')
model = pickle.load(f)

model

f.close()

## Load the test vectors
import h5py

# to verify script - sample cases
#db = h5py.File('./output/dry_run/cats_and_dogs_test_features.hdf5', 'r')

# full dataset
db = h5py.File('./output/cats_and_dogs_test_features.hdf5', 'r')

print(db['features'].shape)

## Do Prediction
pred = model.predict(db['features'])

pred_proba = model.predict_proba(db['features'])

## Generate submission file.
print("Generate submission file...")

pred_file = open("./submission.csv", "w")
file_prob = open("./submission_prob_dog.csv", "w")

pred_file.write("id,label\n")
file_prob.write("id,label\n")

for i in range(0, len(db['ID'])):
    #get the ID
    id = db['ID'][i]
    #print(id, pred[i])
    
    # write the predictions
    pred_file.write('%d,' % id)
    pred_file.write('%d \n' % pred[i]);
    
    #write the prediction proba
    file_prob.write('%d,' % id)
    file_prob.write('%1.4f \n' % pred_proba[i,1])
    
#close the streams.
pred_file.close()
file_prob.close()

print('Done!')
