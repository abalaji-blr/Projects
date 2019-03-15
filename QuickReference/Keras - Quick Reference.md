# Keras - Quick Reference

[TOC]

## Visualization

### How to plot neural network architecture model?

```python
!pip install graphviz

from keras.utils import plot_model
plot_model(model, show_shapes=True, to_file='model2.png')
```



## Model

### How to save model?

### How to load model?



### How to save best model during training?

* Use ModelCheck point

``` python
## to save the best model 
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, 		save_best_only=True)

callbacks = [checkpoint]

model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks)
```



## Google Colab

### How to mount google drive?

```python
# mount google drive
from google.colab import drive

drive.mount('/content/gdrive')  ## After authentication, the google drive will be mounted at the mentioned location.
```



### How to copy file from/to google drive?

```python
# mount the google drive as mentioned above.

# then issue use unix command.
#copy from google drive to local colab machine drive.
!cp '/content/gdrive/My Drive/My Folder/my_file.txt' /content

# you can do vice-versa as well
```



### How to copy file from colab machine to local disk?

```python
from google.colab import files

files.download('model2.png')
```



