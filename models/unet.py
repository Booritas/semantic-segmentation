import keras
from keras import layers

class Unet:
    def __init__(self, input_shape, padding='same'):
        self.input_shape = input_shape
        self.padding = padding
        
    def contracting_block(self, filters, input_layer, name, maxpooling=True):
        if maxpooling:
            maxpool = layers.MaxPool2D(pool_size=2, name=name+"_maxpool")(input_layer)
        else:
            maxpool = input_layer
        conv = layers.Conv2D(filters=filters, kernel_size=3, activation='relu', padding=self.padding, name = name + "_conv_1")(maxpool)
        conv = layers.Conv2D(filters=filters, kernel_size=3, activation='relu', padding=self.padding, name = name + "_conv_2")(conv)
        return conv
    
    def upscaling(self, input_layer, name):
        ups = layers.UpSampling2D(size=(2,2), name=name+"_upscaling")(input_layer)
        return ups
    
    def expanding_block(self, filters, input_layer, concat_layer, name):
        ups = self.upscaling(input_layer, name)
        conv = layers.Conv2D(filters=filters, kernel_size=2, activation='relu', padding='same', name=name+"_conv_0")(ups)
        # compute cropping
        size1 = conv.get_shape().as_list()[1]
        size2 = concat_layer.get_shape().as_list()[1]
        if size1 != size2:
            cropping_left = (size2-size1)//2
            cropping_right = size2 - size1 - cropping_left
            cropping = ((cropping_left, cropping_right),(cropping_left, cropping_right))
            # apply cropping
            crop = layers.Cropping2D(cropping, name=name+"_crop")(concat_layer)
            comb = layers.Concatenate(axis=3, name=name+"_concat")([crop,conv])
        else:
            comb = layers.Concatenate(axis=3, name=name+"_concat1")([concat_layer,conv])
        conv = layers.Conv2D(filters=filters, kernel_size=3, activation='relu', padding=self.padding, name=name+"_conv_1")(comb)
        conv = layers.Conv2D(filters=filters, kernel_size=3, activation='relu', padding=self.padding, name=name+"_conv_2")(conv)
        return conv
    def build_model(self):
        input_layer = layers.Input(shape=self.input_shape, name='input')

        conv1 = self.contracting_block(filters=64,   input_layer=input_layer, name="CNT_1", maxpooling=False)
        conv2 = self.contracting_block(filters=128,  input_layer=conv1, name="CNT_2")
        conv3 = self.contracting_block(filters=256,  input_layer=conv2, name="CNT_3")
        conv4 = self.contracting_block(filters=512,  input_layer=conv3, name="CNT_4")
        conv5 = self.contracting_block(filters=1024, input_layer=conv4, name="CNT_5")
        
        exp1 = self.expanding_block(filters=512, input_layer=conv5, concat_layer=conv4, name="EXP_1")
        exp2 = self.expanding_block(filters=256, input_layer=exp1,  concat_layer=conv3, name="EXP_2")
        exp3 = self.expanding_block(filters=128, input_layer=exp2,  concat_layer=conv2, name="EXP_3")
        exp4 = self.expanding_block(filters=64,  input_layer=exp3,  concat_layer=conv1, name="EXP_4")

        conv = layers.Conv2D(filters=2, kernel_size=3, activation = 'relu', padding = 'same', name="OUT_3")(exp4)
        output_layer = layers.Conv2D(filters=1, kernel_size=1, activation = 'sigmoid', name="OUT")(conv)
        model = keras.Model(input_layer,output_layer)
        return model        