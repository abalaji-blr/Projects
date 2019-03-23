# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

'''
    Standard ResNet50 is too deep for Tiny ImageNet dataset.
    Let's build a custom one.

    1) Instead of 7x7 filter use 5,5 with stride 1
    2) Experiment about removing MaxPool in the entry block
       Reason: Don't want to reduce the input dimension.
    3) About repetition resuidal block - remove last set.
       Reason: it's too deep for this dataset.
       Also, we want to go beyond the object and identify scenes as well.
    4) About kernel count - experiment the following
        * [64, 128, 256]
        * or [128, 256, 512]
        * Make sure the params are well within the limit
    4) Remove Dense Layer
        * Use GAP
        * or 1x1 Conv.
        * By default, network uses 'Avg pool' before flatten.
          Do we need that if we go with GAP?
    5) Stretch goal: Experiment with SeparableConv2D() instead of Conv2D().
'''
class CustomResNet:
    @staticmethod
    def entry_block(input, num_filters):
        '''
        1. In the entry block, use filter 5x5 with stride 1 instead of 7x7.
        2. Optionally think about ignoring the Maxpool in the entry block.
        Idea is, input 64x64 => before the resuidal block, retain same input
        shape or reduced to just 32x32.
        '''
        ## CONV -> BN -> ACT -> MaxPool
        x = Conv2D(num_filters, (5,5), padding='same')(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D((1,1))(x) # next one is max pool, do prep the data accordingly.
        block_output = MaxPooling2D((3,3), strides=(2,2))(x)
        return block_output

    '''
    1x1 CONV -> 3x3 CONV -> 1x1 CONV
    '''
    @staticmethod
    def residual_block(input_data, num_filters, stride):

        short_cut = input_data

        # BN -> Act -> Conv2D(1,1)
        bn1 = BatchNormalization()(input_data)
        act1 = Activation('relu')(bn1)
        conv1 = Conv2D(int(num_filters * 0.25), (1,1))(act1)

        # BN -> Act -> Conv2D(3,3)
        bn2 = BatchNormalization()(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv2D(int(num_filters * 0.25), (3,3), strides=stride, padding='same')(act2)

        # BN -> Act -> Conv2D(1,1)
        bn3 = BatchNormalization()(conv2)
        act3 = Activation('relu')(bn3)
        conv3 = Conv2D(num_filters, (1,1))(act3)

        # need to add with short cut.
        x = add([conv3, short_cut])
        return x

    '''
    Entry block of the residual network, which needs to reduce
    the spatial volume. Basically, need to create the CONV layer
    in the short_cut-branch to match the dimension from the main-branch.
    '''
    @staticmethod
    def residual_block_reduce(input_data, num_filters, stride):
        short_cut = input_data

        # BN -> Act -> Conv2D(1,1)
        bn1 = BatchNormalization()(input_data)
        act1 = Activation('relu')(bn1)
        conv1 = Conv2D(int(num_filters * 0.25), (1, 1))(act1)

        # BN -> Act -> Conv2D(3,3)
        bn2 = BatchNormalization()(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv2D(int(num_filters * 0.25), (3, 3),
                       strides=stride, padding='same')(act2)

        # BN -> Act -> Conv2D(1,1)
        bn3 = BatchNormalization()(conv2)
        act3 = Activation('relu')(bn3)
        conv3 = Conv2D(num_filters, (1, 1))(act3)

        ## to match the dimension from the main-branch with short-cut-branch
        ## apply Conv 1,1 in the short-cut branch, so addition of layers is possible.
        short_cut = Conv2D(num_filters, (1,1), strides=stride)(act1)

        # need to add with short cut.
        x = add([conv3, short_cut])
        return x

    @staticmethod
    def build_resnet(width, height, depth, num_classes, stages, filters):
        """Builds a custom ResNet like architecture.
            Returns:
            The keras `Model`.
        """
        # get the input shape, channels last.
        inputShape = (width, height, depth)

        # get the input layer
        inputs = Input(shape=inputShape)

        ## before apply CONV, it's better to normalize the input
        ## use BatchNorm, by default, channel last
        x = BatchNormalization()(inputs)

        ## build entry block
        x = CustomResNet.entry_block(x, filters[0])

        ## build the residual blocks for different stages
        for i in range(0, len(stages)):
            stride = (1,1) if i == 0 else (2,2)

            # first block need to reduce the spatial size of input vol.
            # in the short-cut-branch to enable concatenation.
            x = CustomResNet.residual_block_reduce(x, filters[i+1], stride)

            # depth of the residual block.
            for j in range(0, stages[i]-1):
               x = CustomResNet.residual_block(x, filters[i+1], (1,1))
        
        ## BN -> Act -> Ave. pooling
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((8,8))(x)

        ## outputs
        x = Flatten()(x)
        x = Dense(num_classes)(x)
        outputs = Activation('softmax')(x)

        ## build the model
        model = Model(inputs, outputs, name='custom_resnet')
        return model

