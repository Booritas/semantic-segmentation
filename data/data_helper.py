import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

def show_empty_masks(image_dir, mask_dir, rows, cols, figsize=(20,10)):
    image_names = os.listdir(image_dir)
    mask_names = os.listdir(mask_dir)
    image_names = [name for name in image_names if os.path.isfile(os.path.join(image_dir, name))]
    image_names.sort()
    plt.figure(figsize=figsize,dpi=80)
    index = 0
    for image_name in image_names:
        image = cv2.imread(os.path.join(image_dir, image_name), 0)
        mask = cv2.imread(os.path.join(mask_dir, image_name), 0)
        if np.max(mask)<1:
            name_parts = image_name.split('_')
            title = name_parts[0][:5] + '...' + name_parts[1]
            data_row = index//cols
            if data_row>=rows:
                break
            data_col = index - (data_row*cols)
            image_row = data_row*2
            mask_row = image_row + 1
            plt.subplot2grid((rows*2,cols),(image_row, data_col))
            plt.imshow(image, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title(title)
            plt.subplot2grid((rows*2,cols),(mask_row, data_col))
            plt.imshow(mask,cmap='gray')
            plt.xticks([])
            plt.yticks([])
            index += 1
    plt.tight_layout()
    plt.show()
    
def show_data(dataset, rows, cols, figsize=(20,10)):
    generator = dataset.generator()
    show_data_from_generator(generator, rows, cols, figsize)
    
def show_data_from_generator(generator, rows, cols, figsize=(20,10)):
    plt.figure(figsize=figsize,dpi=80)
    index = 0
    for data in generator:
        name = None
        image = data[0]
        mask = data[1]
        if len(data)>2:
            name = data[2]
        data_row = index//cols
        if data_row>=rows:
            break
        data_col = index - (data_row*cols)
        image_row = data_row*2
        mask_row = image_row + 1
        plt.subplot2grid((rows*2,cols),(image_row, data_col))
        if len(image.shape)==4:
            image = image[0,:,:,0]
        plt.imshow(image, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if name is not None:
            plt.title(name[:8] + '...' + name[-3:])
        plt.subplot2grid((rows*2,cols),(mask_row, data_col))
        if len(mask.shape)==4:
            mask = mask[0,:,:,0]
        plt.imshow(mask,cmap='gray')
        plt.title(str(mask.shape[0]) + 'x' + str(mask.shape[1]))
        plt.xticks([])
        plt.yticks([])
        index += 1
    plt.tight_layout()
    plt.show()    