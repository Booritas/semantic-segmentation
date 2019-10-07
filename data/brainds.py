import wget
import os
from dataset import Dataset
import zipfile
import glob
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

class BrainTumorDataset(Dataset):
    def __init__(self, dataset_path):
        # url for the dataset
        self.url = "https://ndownloader.figshare.com/articles/1512427/versions/5"
        # root directory where the dataset resides
        self.root_directory = dataset_path
        # directory for original raw data
        self.directory_original = os.path.join(self.root_directory, "original")
        self.directory_train = os.path.join(self.root_directory, "train")
        self.directory_test = os.path.join(self.root_directory, "test")
        self.directory_valid = os.path.join(self.root_directory, "validation")
        self.max_file_index = 3065
        self.image_dir_name = "images"
        self.mask_dir_name = "masks"
    def download(self):
        '''
        Downloads data from the dataset url to directory for the original raw data.
        Extract all files from the downloaded zip file.
        Removes all intermediate files.
        '''
        # path to the dataset file (downloaded)
        data_file_path = os.path.join(self.directory_original, "brain_tumor.zip")
        # create directory for the dataset
        if not os.path.exists(self.directory_original):
            os.makedirs(self.directory_original)
        # download dataset file if needed
        if not os.path.exists(data_file_path):
            wget.download(self.url, data_file_path)
        # unzip the dataset file
        with zipfile.ZipFile(data_file_path) as fl:
            fl.extractall(self.directory_original)
        # remove the dataset file
        if os.path.exists(data_file_path):
            os.remove(data_file_path)
        # unzip child zip files
        zip_pattern = self.directory_original + "/*.zip"
        for file_path in glob.iglob(zip_pattern):
            with zipfile.ZipFile(file_path) as fl:
                fl.extractall(self.directory_original)
        # remove child zip files
        for file_path in glob.iglob(zip_pattern):
            os.remove(file_path)
            
    def clear(self):
        '''
        Removes the dataset directory
        '''
        if os.path.exists(self.root_directory):
            shutil.rmtree(self.root_directory)
            
    def load_raw_data_file(self, path):
        file = h5.File(path,"r")
        data = {}
        data['path'] = path
        data['image']=np.mat(file['/cjdata/image'])
        data['PID'] = np.array2string(np.array(file['/cjdata/PID']).flatten())
        data['label'] = int(np.array(file['/cjdata/label']).flatten()[0])
        data['tumorBorder']=np.mat(file['/cjdata/tumorBorder'])
        data['tumorMask']=np.mat(file['/cjdata/tumorMask'])
        return data

    def raw_data_generator(self, start=1):
        '''
        Returns a generator for the row data
        '''
        directory = self.directory_original
        for index in range(start,self.max_file_index):
            path = f"{directory}/{index}.mat"
            data = self.load_raw_data_file(path)
            if data['image'].shape[0] == 512:
                yield data
            
    def draw_tumor_border(self, image, border):
        img = image.copy()
        vertices = border.reshape((-1,2))
        nv = vertices.shape[0]
        white = int(img.max())
        for v in range(1,nv):
            x1 = int(vertices.item(v-1,0))
            y1 = int(vertices.item(v-1,1))
            x2 = int(vertices.item(v,0))
            y2 = int(vertices.item(v,1))
            img = cv2.line(img,(y1,x1),(y2,x2), white, 2)
        return img
    
    def show_raw_data(self, rows, start=1, picture_size=6, label=None):
        data_gen = self.raw_data_generator(start)
        row = 0
        shape = (rows, 3)
        plt.figure(figsize=(picture_size*shape[1], picture_size*shape[0]))
        for data in data_gen:
            image = data['image']
            mask = data['tumorMask']
            lbl = data['label']
            if label is None or lbl==label:
                plt.subplot2grid(shape, (row, 0))
                plt.xticks([])
                plt.yticks([])
                plt.title(data['PID'])
                plt.imshow(image,cmap='gray')
                plt.subplot2grid(shape, (row, 1))
                plt.xticks([])
                plt.yticks([])
                plt.title(str(data['label']))
                plt.imshow(mask, cmap='gray')
                plt.subplot2grid(shape, (row,2))
                plt.xticks([])
                plt.yticks([])
                image2 = self.draw_tumor_border(image, data['tumorBorder'])
                plt.imshow(image2, cmap='gray')
                row += 1
                if row>=rows:
                    break    
                    
    def show_all_data(self, image_path, picture_size=1, columns=10):
        data_gen = self.raw_data_generator(1)
        rows = (self.max_file_index-1) // columns + 1
        index = 0
        shape = (rows, columns)
        plt.figure(figsize=(picture_size*shape[1], picture_size*shape[0]))
        for data in data_gen:
            row = index//columns
            column = index - row*columns
            image = data['image']
            mask = data['tumorMask']
            lbl = data['label']
            plt.subplot2grid(shape, (row, column))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image,cmap='gray')
            index += 1
        plt.savefig(image_path)

    def split_by_tumor_type(self):
        '''
        returns a dictionary label:[list of file paths]
        '''
        data_gen = self.raw_data_generator(1)
        ds = {1:[],2:[],3:[]}
        
        for item in data_gen:
            label = item['label']
            ds[label].append(item['path'])
        return ds
    
    def split_for_training(self, test_frac=0.1, valid_frac=0.1):
        '''
        splits raw data to training, test and validation datasets
        returns 3 lists with file paths for the training, test and 
        validation sets correspondingly
        '''
        ds = self.split_by_tumor_type()
        train_set = []
        test_set = []
        valid_set = []
        for tumor_type in ds:
            files = ds[tumor_type]
            n_files = len(files)
            n_validate = int(n_files * valid_frac)
            n_test = int(n_files * test_frac)
            n_train = n_files - n_test - n_validate
            train, test, validate = np.split(random.sample(files,n_files), [n_train, n_train + n_test])
            train_set.extend(train)
            test_set.extend(test)
            valid_set.extend(validate)
        return train_set, test_set, valid_set
    
    def prepare_dataset(self, paths, directory):
        image_dir = os.path.join(directory, self.image_dir_name)
        mask_dir = os.path.join(directory, self.mask_dir_name)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        for path in paths:
            file_name = os.path.split(path)[-1].split('.')[0]
            image_path = os.path.join(image_dir, file_name)
            mask_path = os.path.join(mask_dir, file_name)
            data = self.load_raw_data_file(path)
            image = data['image']
            mask = data['tumorMask']
            max_val = image.max()
            min_val = image.min()
            abs_max = max_val - min_val
            image = (image.astype(np.float) - min_val)/abs_max
            np.save(image_path, image)
            np.save(mask_path, mask)
        
    def prepare(self, test_frac=0.1, valid_frac=0.1):
        train, test, valid = self.split_for_training(test_frac, valid_frac)
        self.prepare_dataset(train, self.directory_train)
        self.prepare_dataset(test, self.directory_test)
        self.prepare_dataset(valid, self.directory_valid)
        
    def generator(self, mode='training', batch_size=1):
        directory = ""
        if mode == 'training':
            directory = self.directory_train
        elif mode == 'test':
            directory = self.directory_test
        elif mode == 'validation':
            directory = self.directory_valid
        else:
            raise Exception("Unknown mode")
        image_dir = os.path.join(directory, self.image_dir_name)
        mask_dir = os.path.join(directory, self.mask_dir_name)
        image_files = [file for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]
        count = 0
        images = []
        masks = []
        for name in image_files:
            image = np.load(os.path.join(image_dir, name))
            mask = np.load(os.path.join(mask_dir, name))
            images.append(image)
            masks.append(mask)
            count += 1
            if count == batch_size:
                count = 0
                npim = np.array(images).reshape((batch_size, 512, 512, 1))
                npmsk = np.array(masks).reshape((batch_size, 512, 512, 1))
                images = []
                masks = []
                yield(npim, npmsk)
        
    def show_training_data(self, mode='training', rows=10, picture_size=6):
        data_gen = self.generator(mode)
        row = 0
        shape = (rows, 2)
        plt.figure(figsize=(picture_size*shape[1], picture_size*shape[0]))
        for data in data_gen:
            image = data[0]
            mask = data[1]
            plt.subplot2grid(shape, (row, 0))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image.reshape((512,512)),cmap='gray')
            plt.subplot2grid(shape, (row, 1))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(mask.reshape((512,512)), cmap='gray')
            row += 1
            if row>=rows:
                break    
