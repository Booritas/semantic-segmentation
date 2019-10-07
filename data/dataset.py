class Dataset:

    def __init__(self):
        pass
    
    def normalize_image(image):
        image = image / 255
        return image

    def make_binary(image):
        image[image > 0.5] = 1
        image[image <= 0.5] = 0
        return image
    
    def normalize_label(image):
        image = Dataset.normalize_image(image)
        return Dataset.make_binary(image)
    
    def make_rgb_image(data):
        image = (((data - data.min()) * 255) / (data.max() - data.min())).astype(np.uint8)
        return image

