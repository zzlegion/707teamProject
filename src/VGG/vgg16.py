# encoding: utf-8
from keras.models import Model
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import CSVLogger
import os

# 训练的batch_size
batch_size = 64
# 训练的epoch
epochs = 1000
# 数据位置
TRAIN_DIR = "../../data/sun_train'"
VALID_DIR = "../../data/sun_val"

if __name__ == "__main__":
    # 训练数据
    # train_datagen = ImageDataGenerator(width_shift_range=0.05, height_shift_range=0.05, vertical_flip=True)
    # train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224, 224), batch_size=batch_size, class_mode="binary")
    # image_numbers = train_generator.samples
    # print(train_generator.class_indices)

    # 生成测试数据
    # test_datagen = ImageDataGenerator()
    # validation_generator = test_datagen.flow_from_directory(VALID_DIR, target_size=(224, 224), batch_size=batch_size, class_mode="binary")

    # 使用VGG16的结构，不包括最后一层，且加载ImageNet的预训练参数
    base_model = VGG16(weights=None, include_top=False, pooling='max')
    # base_model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

    # 构建网络的最后一层，二分类
    predictions = Dense(1, activation='sigmoid')(base_model.output)

    # 定义整个模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 继续上一次的训练
    if os.path.exists("resnet50_best.h5"):
        model.load_weights("resnet50_best.h5")
        print("Successfully loaded resnet50_best.h5")

    # 编译模型
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    early_stopping = EarlyStopping(patience=10)
    check_pointer = ModelCheckpoint('vgg16_best.h5', verbose=1, save_best_only=True)
    model.fit_generator(train_generator, steps_per_epoch=image_numbers // batch_size, epochs=epochs,
                        validation_data=validation_generator, validation_steps=batch_size,
                        callbacks=[early_stopping, check_pointer, csv_logger])
