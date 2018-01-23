
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
from keras.layers import Dense, Dropout, Flatten, Conv2D, Add, Input



batch_size = 64

epochs = 100
TRAIN_DIR = "data/train"
VALID_DIR = "data/val"

def SR():
    origin_input = Input(shape=(224, 224, 3))
    x = Conv2D(64, (9, 9), activation='relu', padding='same', name='sr_conv1')(origin_input)
    x = Conv2D(32, (5, 5), activation='relu', padding='same', name='sr_conv2')(x)
    x = Conv2D(3, (5, 5), activation='relu', padding='same', name='sr_conv3')(x)
    out = Add()([origin_input, x])
    return model

if __name__ == "__main__":
    train_datagen = ImageDataGenerator(width_shift_range=0.05, height_shift_range=0.05, vertical_flip=True)
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224, 224), batch_size=batch_size, class_mode="binary")
    image_numbers = train_generator.samples

    test_datagen = ImageDataGenerator()
    validation_generator = test_datagen.flow_from_directory(VALID_DIR, target_size=(224, 224), batch_size=batch_size, class_mode="binary")

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # for layer in base_model.layers[:5]:
    #     layer.trainable = False

    # Adding custom Layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    if os.path.exists("vgg16_best.h5"):
        model.load_weights("vgg16_best.h5")
        print("Successfully loaded vgg16_best.h5")

    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss="binary_crossentropy", metrics=['accuracy'])

    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    early_stopping = EarlyStopping(patience=10)
    check_pointer = ModelCheckpoint('vgg16_best.h5', verbose=1, save_best_only=True)
    model.fit_generator(train_generator, steps_per_epoch=image_numbers // batch_size, epochs=epochs,
                        validation_data=validation_generator, validation_steps=batch_size,
                        callbacks=[early_stopping, check_pointer, csv_logger])
