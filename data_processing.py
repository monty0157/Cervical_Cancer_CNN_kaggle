import os
from keras.preprocessing import image
import random
import numpy as np

#TRAINING DATA WITH EXIF
Type_1_path_train = './data/train/Type_1/'
Type_1_list_train = os.listdir(Type_1_path_train)
type2_path_train = './data/train/Type_2/'
type2_list_train = os.listdir(type2_path_train)
type3_path_train = './data/train/Type_3/'
type3_list_train = os.listdir(type3_path_train)

#TRAINING DATA WITOHUT EXIF
Type_1_path_train_no_exif = './data/train_no_exif/Type_1/'
Type_1_list_train_no_exif = os.listdir(Type_1_path_train_no_exif)
Type_2_path_train_no_exif = './data/train_no_exif/Type_2/'
Type_2_list_train_no_exif = os.listdir(Type_2_path_train_no_exif)
Type_3_path_train_no_exif = './data/train_no_exif/Type_3/'
Type_3_list_train_no_exif = os.listdir(Type_3_path_train_no_exif)

type1_path_val = './data/validation/Type_1/'
type1_list_val = os.listdir(type1_path_val)
type2_path_val = './data/validation/Type_2/'
type2_list_val = os.listdir(type2_path_val)
type3_path_val = './data/validation/Type_3/'
type3_list_val = os.listdir(type3_path_val)

test_path_train = './data/test/'
test_list_train = os.listdir(test_path_train)

train_directories = ['Type_1', 'Type_2', 'Type_3']
val_directories = ['Type_1', 'Type_3']

#PREPARING DATA FOR GRID SEARCH
def grid_search_helper(target_size):

    images_list = []
    labels_list = []
    for i in range(len(train_directories)):
        for file in eval(train_directories[i] + '_list_train_no_exif'):
            if (file != '.DS_Store'):
                img = image.load_img(eval(train_directories[i] + '_path_train_no_exif') + '/' + file, target_size = target_size)
                img = np.asarray(img)
    
                #RESCALE IMAGE
                img = img/255
                images_list.append(img)
    
                #ADD AS CLASS 0
                labels_list.append(i)

    #SHUFFLE DATA
    zip_data_for_shuffle = list(zip(images_list,labels_list))
    random.shuffle(zip_data_for_shuffle)
    images_list, labels_list = zip(*zip_data_for_shuffle)

    return np.asarray(images_list), np.asarray(labels_list)

def remove_exif():
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    '''for i in range(train_directories):
        for file in train_directories[i]:
            if (file != '.DS_Store'):

                image = Image.open((train_directories[i] + _path_train) + file)

                # next 3 lines strip exif
                data = list(image.getdata())
                image_without_exif = Image.new(image.mode, image.size)
                image_without_exif.putdata(data)

                image_without_exif.save('./data/train_no_exif/' + train_directories[i] + '/' + file)'''

    for file in type2_list_train:
        if (file != '.DS_Store'):

            image = Image.open(type2_path_train + file)

            # next 3 lines strip exif
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)

            image_without_exif.save('./data/train_no_exif/Type_2/' + file)

    '''for file in type3_list_train:
        if (file != '.DS_Store'):

            image = Image.open(type3_path_train + file)

            # next 3 lines strip exif
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)

            image_without_exif.save('./data/train_no_exif/Type_3/' + file)'''

    '''for file in test_list_train:
        if (file != '.DS_Store'):

            image = Image.open(test_path_train + file)

            # next 3 lines strip exif
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)

            image_without_exif.save('./data/test_no_exif/' + file)'''


    '''for file in type1_list_val:
        if (file != '.DS_Store'):

            image = Image.open(type1_path_val + file)

            # next 3 lines strip exif
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)

            image_without_exif.save('./data/validation_no_exif/Type_1/' + file)'''

    for file in type2_list_val:
        if (file != '.DS_Store'):

            image = Image.open(type2_path_val + file)

            # next 3 lines strip exif
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)

            image_without_exif.save('./data/validation_no_exif/Type_2/' + file)

    '''for file in type3_list_val:
        if (file != '.DS_Store'):

            image = Image.open(type3_path_val + file)

            # next 3 lines strip exif
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)

            image_without_exif.save('./data/validation_no_exif/Type_3/' + file)'''
