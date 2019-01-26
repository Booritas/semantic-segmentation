from dataset import Dataset
import numpy as np
import os
import zipfile
import shutil
import cv2
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.cluster import KMeans

class DataScienceBowl2018(Dataset):
    
    def __init__(self, dataset_path):
        self.directory = dataset_path
        self.train_file_path = os.path.join(self.directory,'stage1_train.zip')
        self.data_path = os.path.join(self.directory,'data')
        prepared_data_path = os.path.join(self.directory,'prepared_data')
        self.prepared_image_path = os.path.join(prepared_data_path,'images')
        self.prepared_mask_path = os.path.join(prepared_data_path,'masks')
        self.first_image_folder = os.path.join(self.data_path,'00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e')
        self.broken_image = '7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80'
        self.augmentation = dict(rotation_range=0.2, width_shift_range=0.05,
                                height_shift_range=0.05, shear_range=0.05,
                                zoom_range=0.05, horizontal_flip=True, fill_mode='nearest',
                                featurewise_center = False,
                                samplewise_center = False,
                                featurewise_std_normalization = False,
                                samplewise_std_normalization = False
                                )
        self.tile_shape = (256,256)
    def download(self):
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.exists(self.train_file_path):
            os.system('kaggle competitions download -c data-science-bowl-2018 -p ' + self.directory)
        if not os.path.exists(self.first_image_folder):
            with zipfile.ZipFile(self.train_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_path)
                
    def clear(self):
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
        
    def combine_masks(self, files):
        mask = None
        for file in files:
            pix = cv2.imread(file, 0)
            if mask is None:
                mask = np.zeros_like(pix,dtype=np.uint8)
            mask[pix>0] = 255
        return mask
    
    def calculate_shifts(self, W, w):
        n = (W-1)//w + 1
        dt = 0
        if n > 1:
            dt = (w*n - W)//(n - 1)
            if dt<10:
                n = n + 1
            dt = w - (w*n - W)//(n - 1)
        return dt, n
    
    def split_image_to_tiles(self, image_data):
        W, H = image_data.shape
        w, h = self.tile_shape
        dx, nx = self.calculate_shifts(W, w)
        dy, ny = self.calculate_shifts(H, h)
        image_tiles = []
        for x in range(nx):
            for y in range(ny):
                ix = x*dx
                iy = y*dy
                sx = W - ix
                sy = H - iy
                if sx > w:
                    sx = w
                if sy > h:
                    sy = h
                image_tile = np.zeros(self.tile_shape,dtype=np.uint8)
                image_tile[:sx, :sy] = image_data[ix:ix+sx, iy:iy+sy]
                image_tiles.append(image_tile)
        return image_tiles
    
    def combine_image_from_tiles(self, image_shape, image_tiles):
        H, W = image_shape
        h, w = image_tiles[0].shape
        col_shift, col_shifts = self.calculate_shifts(W, w)
        row_shift, row_shifts = self.calculate_shifts(H, h)

        image = np.zeros(image_shape)
        for index, image_tile in enumerate(image_tiles):
            col_tile = index%col_shifts
            row_tile = index//col_shifts
            col = col_shift*col_tile
            row = row_shift*row_tile
            cols = W - col
            rows = H - row
            if cols > w:
                cols = w
            if rows > h:
                rows = h
            image[row:row+rows, col:col+cols] = image_tile[:rows, :cols]
        return image
    
    def save_prepared_data(self, set_name, image_name, image_tiles, mask_tiles):
        tiles = len(image_tiles)
        for tile in range(tiles):
            if tiles>1:
                file_name = '{name}_{tile}.png'.format(name=image_name, tile=tile) 
            else:
                file_name = image_name + '.png'
            cv2.imwrite(os.path.join(self.get_dataset_image_path(set_name), file_name), image_tiles[tile])
            cv2.imwrite(os.path.join(self.get_dataset_mask_path(set_name), file_name), mask_tiles[tile])
        
    def prepare_image(self, image_name, set_name):
        image_path = self.image_path_from_name(image_name)
        mask_dir = self.get_dataset_mask_path(image_name)
        image_dir = self.get_dataset_image_path(image_name)
        source_mask_dir = os.path.join(self.data_path,image_name,'masks')
        mask_files = [os.path.join(source_mask_dir, mask) for mask in os.listdir(source_mask_dir) if os.path.isfile(os.path.join(source_mask_dir, mask))]
        mask_data = self.combine_masks(mask_files)
        image_data = cv2.imread(image_path, 0)
        if set_name!='test':
            mean = np.mean(image_data)
            if mean>80:
                image_data = (255 - image_data)
            image_tiles = self.split_image_to_tiles(image_data)
            mask_tiles = self.split_image_to_tiles(mask_data)
        else:
            image_tiles = [image_data]
            mask_tiles = [mask_data]
        self.save_prepared_data(set_name, image_name, image_tiles, mask_tiles)
        
    def get_dataset_path(self, set_name):
        return os.path.join(self.directory, set_name)
    
    def get_dataset_image_path(self, set_name):
        return os.path.join(self.directory, set_name, 'images')
    
    def get_dataset_mask_path(self, set_name):
        return os.path.join(self.directory, set_name, 'masks')
    
    def prepare_dataset(self, image_names, set_name):
        '''
        prepare a directory for a dataset with images
        and masks
        '''
        image_dir = self.get_dataset_image_path(set_name)
        mask_dir = self.get_dataset_mask_path(set_name)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        for image_name in image_names:
            self.prepare_image(image_name, set_name)
            
    def prepare(self, test_frac=0.1, valid_frac=0.1):
        image_lists = self.split_dataset(cluster_count=8)
        dataset_names = ['training','validation','test']
        for dataset in zip(dataset_names, image_lists):
            dataset_path = self.get_dataset_path(dataset[0])
            self.prepare_dataset(dataset[1], dataset[0])
    
    def generator(self, mode='training', batch_size=1):
        '''
        Returns a generator for batches of images and corresponded labels.
        Generator yields a pair (image_batch[batch_size, image_height, image_width, channels], 
        label_batch[batch_size, label_height, label_width, 1]),
        normalized to [0.0, 1.0] range.
        Training set uses image augmentation. Validation and test sets do
        not use any augmentation. 
        '''
        if mode=='test':
            image_dir = self.get_dataset_image_path('test')
            mask_dir = self.get_dataset_mask_path('test')
            image_names = [name for name in os.listdir(image_dir)]
            for image_name in image_names:
                image_path = os.path.join(image_dir, image_name)
                if os.path.isfile(image_path):
                    mask_path = os.path.join(mask_dir, image_name)
                    image = cv2.imread(image_path, 0)
                    mask = cv2.imread(mask_path, 0)
                    yield (image, mask)
        else:
            augmentation = {}

            indices = None
            if mode=='training':
                augmentation = self.augmentation

            image_gen = ImageDataGenerator(augmentation)
            label_gen = ImageDataGenerator(augmentation)

            seed = 1

            image_path = self.get_dataset_image_path(mode)
            mask_path = self.get_dataset_mask_path(mode)

            image_root, image_dir = os.path.split(image_path)
            mask_root, mask_dir = os.path.split(mask_path)

            image_generator = image_gen.flow_from_directory(image_root, classes=[image_dir], color_mode='grayscale', 
                                                            batch_size = batch_size, seed = seed, target_size=(256, 256),
                                                            class_mode=None)

            label_generator = label_gen.flow_from_directory(mask_root, classes=[mask_dir], color_mode='grayscale',
                                                            batch_size = batch_size, seed = seed,
                                                            target_size=(256, 256), class_mode=None)

            data_generator = zip(image_generator, label_generator)
            for (image, label) in data_generator:
                yield (Dataset.normalize_image(image), Dataset.normalize_label(label))

    def source_generator(self):
        image_names = self.get_image_names()
        return self.generator_from_names(image_names)
    
    def generator_from_names(self, image_names):
        for image_name in image_names:
            if image_name!=self.broken_image:
                image_path = os.path.join(self.data_path, image_name,'images', image_name + '.png')
                mask_dir = os.path.join(self.data_path, image_name, 'masks')
                mask_files = [os.path.join(mask_dir, mask) for mask in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, mask))]
                mask_data = self.combine_masks(mask_files)
                image_data = cv2.imread(image_path, 0)
                yield (image_data, mask_data, image_name)
    
    def create_image_clusters(self, df, cluster_count=8):
        '''
        executes k-means algorithm to split the whole dataset in
        "cluster_count" clusters according to raster parameters
        of images
        '''
        columns = list(df.columns.values)
        df0 = df[columns[1:]]
        kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(df0)
        df['cluster'] = kmeans.labels_
        clusters = df.cluster.max() + 1
        
    def get_image_names(self):
        '''
        returns a list of image names in the dataset
        '''
        return [dir for dir in os.listdir(self.data_path) if dir!=self.broken_image and os.path.isdir(os.path.join(self.data_path, dir))]
    
    def image_path_from_name(self, image_name):
        '''
        returns image path constructed from the image name
        '''
        return os.path.join(self.data_path, image_name, 'images', image_name + '.png')
    
    def compute_image_stats(self):
        '''
        Creates a pandas DataFrame object. Each row contains image name
        and raster statistics corresponded to the image
        '''
        data_path = self.data_path
        image_names = self.get_image_names()
        rows = []
        for image_name in image_names:
            image_path = self.image_path_from_name(image_name)
            image = cv2.imread(image_path, 0)
            image = image/255.
            row = (image_name, np.min(image), np.max(image), np.std(image), np.mean(image), image.shape[0]/1024, image.shape[1]/1024)
            rows.append(row)
        columns = ['name','min','max','std','mean','width','height']
        df = pd.DataFrame.from_records(rows, columns=columns)
        return df
    
    def split_dataset(self, test_frac=0.1, valid_frac=0.1, cluster_count=8):
        '''
        Splits the whole dataset in 3 subsets: training set, validation set and test set.
        Parameters:
            test_frac: fraction of test dataset in the whole data set
            valid_frac: fraction of validation dataset in the whole data set
            cluster_count: number of clusters. The whole dataset will be splitted in
                clusters according to the image parameters. Each cluster will be splitted
                in subsets separately in order to make splits more homogeneous.
        The method returns 3 lists with names of images.
        '''
        df = self.compute_image_stats()
        self.create_image_clusters(df, cluster_count)
        test_and_valid_df = None
        frac = valid_frac + test_frac
        for cluster in range(cluster_count):
            df0 = df[df.cluster==cluster]
            df1 = df0.sample(frac=frac)
            if test_and_valid_df is None:
                test_and_valid_df = df1
            else:
                test_and_valid_df = test_and_valid_df.append(df1)
        test_frac1 = test_frac/frac
        test_df = test_and_valid_df.sample(frac=test_frac1)
        all_names = df.name.tolist()
        test_names = test_df.name.tolist()
        test_and_valid_names = test_and_valid_df.name.tolist()
        valid_names = [name for name in test_and_valid_names if name not in test_names]
        train_names = [name for name in all_names if name not in test_and_valid_names]
        return train_names, valid_names, test_names