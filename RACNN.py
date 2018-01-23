# encoding: utf-8
from keras.models import Model
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import CSVLogger
import os
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, Add, Input, MaxPooling2D

batch_size = 64
epochs = 100

RACNN_ALL_DATA_ROOT = "sundata/all/"


def network():
    img_input = Input(shape=(224, 224,3))
    # SR
    x = Conv2D(64, (9, 9), activation='relu', padding='same', name='sr_conv1')(img_input)
    x = Conv2D(32, (5, 5), activation='relu', padding='same', name='sr_conv2')(x)
    x = Conv2D(3, (5, 5), activation='relu', padding='same', name='sr_conv3')(x)
    out = Add()([img_input, x])

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(out)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # Create model.
    model = Model(img_input, predictions, name='srvgg16')
    return model


def loadsr_weight(sr_model):
    return sr_model


def loadimagenet_weight(sr_model):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer_idx in range(len(base_model.layers)):
        sr_model.layers[layer_idx+4].set_weights(base_model.layers[layer_idx].get_weights())
    return sr_model


def train(DATA_ROOT, exp_name):
    TRAIN_DIR = DATA_ROOT + "train"
    VALID_DIR = DATA_ROOT + "val"
    print "training on " + DATA_ROOT

    sr_model = network()
    sr_model = loadsr_weight(sr_model)
    sr_model = loadimagenet_weight(sr_model)

    for layer in sr_model.layers[4:14]:
        layer.trainable = False

    train_datagen = ImageDataGenerator(width_shift_range=0.05, height_shift_range=0.05, rotation_range=180)
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224, 224), batch_size=batch_size,
                                                        class_mode="binary")
    image_numbers = train_generator.samples
    test_datagen = ImageDataGenerator()
    validation_generator = test_datagen.flow_from_directory(VALID_DIR, target_size=(224, 224), batch_size=batch_size,
                                                            class_mode="binary")

    if os.path.exists("vgg16_"+exp_name+"_best.h5"):
        sr_model.load_weights("vgg16_"+exp_name+"_best.h5")
        print("Successfully loaded "+"vgg16_"+exp_name+"_best.h5")

    sr_model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss="binary_crossentropy", metrics=['accuracy'])

    csv_logger = CSVLogger(exp_name+'log.csv', append=True, separator=';')
    early_stopping = EarlyStopping(patience=10)
    check_pointer = ModelCheckpoint("vgg16_"+exp_name+"_best.h5", verbose=1, save_best_only=True)
    sr_model.fit_generator(train_generator, steps_per_epoch=image_numbers // batch_size, epochs=epochs,
                        validation_data=validation_generator, validation_steps=batch_size,
                        callbacks=[early_stopping, check_pointer, csv_logger])

if __name__ == "__main__":
    train(RACNN_ALL_DATA_ROOT, "sun_RACNN_all_f5_14")
