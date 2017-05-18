
#IMPORT LIBRARIES
import keras
import numpy as np
import pandas as pd

#BUILD CNN
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout,  GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16

'''base_model = VGG16(include_top = False, input_shape = (64,64, 3))


# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(100, activation='relu')(x)
x = Dense(100, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

#for layer in base_model.layers:
#    layer.trainable = False

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])'''


def build_model(optimizer = 'rmsprop', units = 128, filters = 32, kernel_size = 3, dropout_layers = 0):
    model = Sequential()
    model.add(Convolution2D(filters = filters,
                            kernel_size = kernel_size,
                            strides = 1,
                            padding = 'same',
                            activation = 'relu',
                            input_shape = (100, 100, 3)))

    model.add(MaxPooling2D(strides = 2, pool_size = 2))
    model.add(Convolution2D(filters = filters,
                        kernel_size = kernel_size,
                        strides = 1,
                        padding = 'same',
                        activation = 'relu'))

    model.add(MaxPooling2D(strides = 2, pool_size = 2))
    model.add(Flatten())

    model.add(Dense(units = units, activation = 'relu'))
    if(dropout_layers >= 1):
        model.add(Dropout(rate = 0.2))
    model.add(Dense(units = units, activation = 'relu'))

    if(dropout_layers >= 2):
        model.add(Dropout(rate = 0.2))
    model.add(Dense(units = 3, activation = 'softmax'))

    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

#CREATING TEST SET AND TRAINING GENERATORS
from keras.preprocessing.image import ImageDataGenerator
batch_size = 32
n_images = 4808
n_images_validation = 1455
n_images_test = 512
model = build_model()

train_datagen = ImageDataGenerator(rescale = 1./255)
                                   #rotation_range = 0.2,
                                   #zoom_range = 0.2,
                                   #horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)


train_dataset = train_datagen.flow_from_directory('data/train_no_exif',
                                                  batch_size = batch_size,
                                                  class_mode = 'categorical',
                                                  target_size = (100,100))

validation_dataset = test_datagen.flow_from_directory('data/validation_no_exif',
                                                  batch_size = batch_size,
                                                  class_mode = 'categorical',
                                                  target_size = (100,100))

test_dataset = test_datagen.flow_from_directory('data/test_no_exif',
                                                  batch_size = batch_size,
                                                  class_mode = 'categorical',
                                                  target_size = (100,100),
                                                  shuffle = False)


#TRAINING AND EVALUATING ON IMAGE GENERATOR
from keras.callbacks import ModelCheckpoint
save_model = ModelCheckpoint('model.{epoch:02d}--{val_loss:.2f}_own_model_500.hdf5', monitor = 'val_loss', save_best_only = True)

model.fit_generator(train_dataset,
                            steps_per_epoch = n_images/batch_size,
                            epochs = 100,
                            validation_data = validation_dataset,
                            validation_steps = n_images_validation,
                            workers = 32,
                            callbacks=[save_model])

#model.load_weights('model.12--0.86_own_model_500.hdf5')
#prediction = model.predict_generator(test_dataset, steps=n_images_test/batch_size)
#print(prediction)
#pd.DataFrame(prediction, columns=['Type_1', 'Type_2', 'Type_3']).to_csv('prediction_own_model_2.csv')
print('Accuracy, loss:', model.evaluate_generator(validation_dataset, steps = n_images_validation/batch_size))

'''#GRIDSEARCH
from data_processing import grid_search_helper
images_list, labels_list = grid_search_helper(target_size = (224,224))

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
grid_search = KerasClassifier(build_fn = model)

parameters = {
    'dropout_layers': [0, 1, 2],
    'epochs': [25, 50],
    'units': [75, 200],
    'batch_size': [25, 32],
}

grid_search = GridSearchCV(estimator = grid_search, param_grid = parameters, scoring = 'accuracy', cv = 3)
grid_search = grid_search.fit(images_list, labels_list)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print('best_parameters:', best_parameters, 'Accuracy:', best_accuracy)
parameters = grid_search.cv_results_
print(parameters)


#TESTING ON SINGLE IMAGE
from keras.preprocessing import image
image_open = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
image_array = np.asarray(image_open, dtype="uint8")
image_array = np.expand_dims(image_array, axis = 0)

result = model.predict(image_array)
class_index = train_dataset.class_indices
print(result, class_index)'''
