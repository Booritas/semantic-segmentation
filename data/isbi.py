import sys
from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from dataset import Dataset

class ISBI_2012(Dataset):
    
    def __init__(self, dataset_path='./datasets/isbi-2012'):
        self.images = os.path.join(dataset_path,'train-volume.tif')
        self.labels = os.path.join(dataset_path,'train-labels.tif')
        self.validation_set = {5, 15, 25}
        self.test_set = {0, 10, 20}
        self.page_count = 30
        all_indices = set([x for x in range(self.page_count)])
        self.training_set = all_indices - self.validation_set - self.test_set
        self.augmentation = dict(rotation_range=0.2, width_shift_range=0.05,
                                height_shift_range=0.05, shear_range=0.05,
                                zoom_range=0.05, horizontal_flip=True, fill_mode='nearest')
        self.image_data = None
        self.label_data = None
    def prepare_data(self):
        if self.image_data is None or self.label_data is None:
            image_tif = Image.open(self.images)
            label_tif = Image.open(self.labels)
            image_shape = (self.page_count, image_tif.size[0], image_tif.size[1], 1)
            label_shape = (self.page_count, label_tif.size[0], label_tif.size[1], 1)
            
            self.image_data = np.empty(image_shape, dtype=float)
            self.label_data = np.empty(label_shape, dtype=float)
            
            for index in range(self.page_count):
                image_tif.seek(index)
                label_tif.seek(index)
                image = np.array(image_tif.getdata()).reshape(image_tif.size[0], image_tif.size[1], 1)
                label = np.array(label_tif.getdata()).reshape(label_tif.size[0], label_tif.size[1], 1)
                self.image_data[index,:,:,:] = Dataset.normalize_image(image)
                self.label_data[index,:,:,:] = Dataset.normalize_label(label)
            
    def generator(self, mode='training', batch_size=1):
        '''
        Returns a generator for batches of images and corresponded labels.
        Generator yields a pair (image_batch[batch_size, image_height, image_width, channels], 
        label_batch[batch_size, label_height, label_width, 1]),
        normalized to [0.0, 1.0] range.
        Training set uses image augmentation. Validation and test sets do
        not use any augmentation. 
        '''
        self.prepare_data()
        
        augmentation = {}
        indices = None
        if mode=='training':
            augmentation = self.augmentation
            indices = self.training_set
        elif mode=='validation':
            indices = self.validation_set
        elif mode=='test':
            indices = self.test_set
        else:
            raise ValueError('Unknown dataset mode ', mode)
            
        number_of_images = len(indices)
        
        image_shape = (number_of_images, self.image_data.shape[1], self.image_data.shape[2], 1)
        label_shape = (number_of_images, self.label_data.shape[1], self.label_data.shape[2], 1)
        
        image_data = np.empty(image_shape, dtype=float)
        label_data = np.empty(label_shape, dtype=float)
        
        for index, page in enumerate(indices):
            image_data[index,:,:,:] = self.image_data[page,:,:,:]
            label_data[index,:,:,:] = self.label_data[page,:,:,:]
            
        image_gen = ImageDataGenerator(**augmentation)
        label_gen = ImageDataGenerator(**augmentation)
        
        image_gen.fit(image_data, augment=True)
        label_gen.fit(label_data, augment=True)
        seed = 1
        image_generator = image_gen.flow(x = image_data, batch_size = batch_size, seed = seed)
        label_generator = label_gen.flow(x = label_data, batch_size = batch_size, seed = seed)
        data_generator = zip(image_generator, label_generator)
        for (image, label) in data_generator:
            yield (image, label)