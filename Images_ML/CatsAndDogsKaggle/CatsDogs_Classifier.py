import h5py

# to verify script - sample cases
#db = h5py.File('./output/dry_run/cats_and_dogs_features.hdf5', 'r')

# full dataset
db = h5py.File('./output/cats_and_dogs_features.hdf5', 'r')

## train : validation split -> 75:25
train_idx = int(db['labels'].shape[0] * 0.75)

print(train_idx)

## Build Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

##
print('Tuning parameters...')
params = { 'C': [0.0001, 0.001, 0.01, 0.1, 1]}

classifier = GridSearchCV(LogisticRegression(), params, cv=3)

classifier.fit(db['features'][:train_idx], db['labels'][:train_idx])

print('Best hyperparameter : {}'.format(classifier.best_params_) )

## Evaluate the model
#do the prediction for the validation set
pred = classifier.predict(db['features'][train_idx:])

pred_proba = classifier.predict_proba(db['features'][train_idx:])

from sklearn.metrics import classification_report

print(classification_report(db['labels'][train_idx:], pred, target_names=['cat', 'dog']))

## Save the Model
import pickle

#classifier.best_estimator_

model_file = './model/cats_and_dogs.pickle'
f = open(model_file, 'wb')
f.write(pickle.dumps(classifier.best_estimator_))
f.close()

# close the hdf5 database
db.close()
