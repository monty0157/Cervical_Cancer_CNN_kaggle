import os
from keras.preprocessing import image
import random
import numpy as np

type1_path_train = './data/train/Type_1/'
type1_list_train = os.listdir(type1_path_train)
type2_path_train = './data/train/Type_2/'
type2_list_train = os.listdir(type2_path_train)
type3_path_train = './data/train/Type_3/'
type3_list_train = os.listdir(type3_path_train)

#PREPARING DATA FOR GRID SEARCH
def grid_search_helper(target_size):

    images_list = []
    labels_list = []
    for file in cats_list_train:
        if (file != '.DS_Store'):
            img = image.load_img(cats_path_train + file, target_size = target_size)
            img = np.asarray(img)

            #RESCALE IMAGE
            img = img/255
            images_list.append(img)

            #ADD AS CLASS 0
            labels_list.append(0)
    for file in dogs_list_train:
        if (file != '.DS_Store'):
            img = image.load_img(dogs_path_train + file, target_size = target_size)
            img = np.asarray(img)

            #RESCALE IMAGE
            img = img/255
            images_list.append(img)

            #ADD AS CLASS 1
            labels_list.append(1)

    #SHUFFLE DATA
    zip_data_for_shuffle = list(zip(images_list,labels_list))
    random.shuffle(zip_data_for_shuffle)
    images_list, labels_list = zip(*zip_data_for_shuffle)

    return np.asarray(images_list), np.asarray(labels_list)

def remove_exif():
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    for file in type1_list_train:
        if (file != '.DS_Store'):

            image = Image.open(type1_path_train + file)
        
            # next 3 lines strip exif
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)
        
            image_without_exif.save('./data/train_no_exif/Type_1/' + file)
            
    for file in type2_list_train:
        if (file != '.DS_Store'):

            image = Image.open(type2_path_train + file)
        
            # next 3 lines strip exif
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)
        
            image_without_exif.save('./data/train_no_exif/Type_2/' + file)
            
    for file in type3_list_train:
        if (file != '.DS_Store'):

            image = Image.open(type3_path_train + file)
        
            # next 3 lines strip exif
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)
        
            image_without_exif.save('./data/train_no_exif/Type_3/' + file)
            
remove_exif()