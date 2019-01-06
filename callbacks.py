import keras
from PIL import Image
import io
import numpy as np
import tensorflow as tf

class ImageHistory(keras.callbacks.Callback):
    
    def __init__(self, tensor_board_dir, data, last_step=0, draw_interval=100):
        self.data = data
        self.last_step = last_step
        self.draw_interval = draw_interval
        self.tensor_board_dir = tensor_board_dir
        
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        images = []
        labels = []
        for item in self.data:
            image_data = item[0]
            label_data = item[1]
            y_pred = self.model.predict(image_data)
            images.append(y_pred)
            labels.append(label_data)
        image_data = np.concatenate(images,axis=2)
        label_data = np.concatenate(labels,axis=2)
        data = np.concatenate((image_data,label_data), axis=1)
        self.saveToTensorBoard(data, 'epoch', epoch)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        if batch % self.draw_interval == 0:
            images = []
            labels = []
            for item in self.data:
                image_data = item[0]
                label_data = item[1]
                y_pred = self.model.predict(image_data)
                images.append(y_pred)
                labels.append(label_data)
            image_data = np.concatenate(images,axis=2)
            label_data = np.concatenate(labels,axis=2)
            data = np.concatenate((image_data,label_data), axis=1)
            self.last_step += 1
            self.saveToTensorBoard(data, 'batch', self.last_step*self.draw_interval)
        return
    
    def make_image(self, npyfile):
        """
        Convert an numpy representation image to Image protobuf.
        taken and updated from https://github.com/lanpa/tensorboard-pytorch/
        """
        height, width, channel = npyfile.shape
        image = Image.frombytes('L',(width,height), npyfile.tobytes())
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height, width=width, colorspace=channel,
                             encoded_image_string=image_string)
    
    def saveToTensorBoard(self, npyfile, tag, epoch):
        data = npyfile[0,:,:,:]
        image = (((data - data.min()) * 255) / (data.max() - data.min())).astype(np.uint8)
        image = self.make_image(image)
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=image)])
        writer = tf.summary.FileWriter(self.tensor_board_dir)
        writer.add_summary(summary, epoch)
        writer.close()        
    