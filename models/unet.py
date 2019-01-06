import keras
from keras import layers

class Unet:
    def __init__(self, input_shape, padding='same'):
        self.input_shape = input_shape
        self.padding = padding
        
    def contracting_block(self, filters, input_layer, maxpooling=True):
        if maxpooling:
            maxpool = layers.MaxPool2D(pool_size=2)(input_layer)
        else:
            maxpool = input_layer
        conv = layers.Conv2D(filters=filters, kernel_size=3, activation='relu', padding=self.padding)(maxpool)
        conv = layers.Conv2D(filters=filters, kernel_size=3, activation='relu', padding=self.padding)(conv)
        return conv
    
    def upscaling(self, input_layer):
        ups = layers.UpSampling2D(size=(2,2))(input_layer)
        return ups
    
    def expanding_block(self, filters, input_layer, concat_layer):
        ups = self.upscaling(input_layer)
        conv = layers.Conv2D(filters=filters, kernel_size=2, activation='relu', padding='same')(ups)
        # compute cropping
        size1 = conv.get_shape().as_list()[1]
        size2 = concat_layer.get_shape().as_list()[1]
        if size1 != size2:
            cropping_left = (size2-size1)//2
            cropping_right = size2 - size1 - cropping_left
            cropping = ((cropping_left, cropping_right),(cropping_left, cropping_right))
            # apply cropping
            crop = layers.Cropping2D(cropping)(concat_layer)
            comb = layers.Concatenate(axis=3)([crop,conv])
        else:
            comb = layers.Concatenate(axis=3)([concat_layer,conv])
        conv = layers.Conv2D(filters=filters, kernel_size=3, activation='relu', padding=self.padding)(comb)
        conv = layers.Conv2D(filters=filters, kernel_size=3, activation='relu', padding=self.padding)(conv)
        return conv
    
    def build_model(self):
        input_layer = layers.Input(shape=self.input_shape, name='input')

        conv1 = self.contracting_block(filters=64,   input_layer=input_layer, maxpooling=False)
        conv2 = self.contracting_block(filters=128,  input_layer=conv1)
        conv3 = self.contracting_block(filters=256,  input_layer=conv2)
        conv4 = self.contracting_block(filters=512,  input_layer=conv3)
        conv5 = self.contracting_block(filters=1024, input_layer=conv4)
        
        exp1 = self.expanding_block(filters=512, input_layer=conv5, concat_layer=conv4)
        exp2 = self.expanding_block(filters=256, input_layer=exp1,  concat_layer=conv3)
        exp3 = self.expanding_block(filters=128, input_layer=exp2,  concat_layer=conv2)
        exp4 = self.expanding_block(filters=64,  input_layer=exp3,  concat_layer=conv1)

        conv = layers.Conv2D(filters=2, kernel_size=3, activation = 'relu', padding = 'same')(exp4)
        output_layer = layers.Conv2D(filters=1, kernel_size=1, activation = 'sigmoid')(conv)
        model = keras.Model(input_layer,output_layer)
        return model        