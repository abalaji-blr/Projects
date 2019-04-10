
## Intel Scene Classifier

## Load the database of Intel Scene features.
import h5py
db = h5py.File('./output/IntelSceneExtractedFeatures.hdf5', 'r')

#print(list(db.keys()))
print(db["features"].shape)

### Build Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

## params for logistic regression
## C - Regularization parameter
params = { 'C': [0.0001, 0.001, 0.01, 0.1, 1]}

classifier = GridSearchCV(LogisticRegression(), params, cv=3)

### Train the model
classifier.fit(db['features'][:], db['labels'][:])
print('Best hyperparameter : {}'.format(classifier.best_params_) )

### Evaluate the model
#from sklearn.metrics import classification_report, confusion_matrix

### Save the Model
import pickle
model_file = './model/IntelSceneClassifier.pickle'

f = open(model_file, 'wb')
f.write(pickle.dumps(classifier.best_estimator_))
f.close()

### Close the database
db.close()
