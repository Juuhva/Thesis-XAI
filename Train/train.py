import os
import cv2
import keras
import numpy as np
import imghdr
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.src.metrics import Recall, Precision, CategoricalAccuracy
import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

data_dir = '../Dataset/train'
image_exts = ['jpg', 'jpeg', 'png', 'bmp']
model_name = 'imageClassifierResEarlyStop.keras'

def remove_wrong_images(data_dir, image_exts):
    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print('Image not in extension list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e:
                print('Issue with image {}'.format(image_path))
                os.remove(image_path)

def load_data(data_dir):
    data = tf.keras.utils.image_dataset_from_directory(data_dir, batch_size=16, image_size=(256, 256))
    data = data.map(lambda x, y: (x / 255, y))
    return data
def split_data(data):
    train_size = int(len(data) * .6)
    val_size = int(len(data) * .2)
    test_size = int(len(data) * .2)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    return train, val, test
def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(filters * 2, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv2D(filters * 2, kernel_size, strides=stride, padding='same')(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.LeakyReLU(alpha=0.05)(x)
    return x
def build_res_model(input_shape=(256, 256, 3), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (7, 7), strides=2, padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    x = residual_block(x, filters=32, stride=1)
    x = residual_block(x, filters=64, stride=2)
    x = residual_block(x, filters=128, stride=2)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def save_trained_model(model, model_path):
    model.save(model_path)

def save_confusion_matrix(model, test):
    y_true = []
    y_pred = []
    class_names = ["cat", "dog", "human"]
    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        yhat = np.argmax(yhat, axis=1)

        y_true.extend(y)
        y_pred.extend(yhat)

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    save_path = "plots/confusion_matrix_" + model_name + ".png"

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Confusion Matrix Normalized")
    plt.savefig(save_path)
    plt.show()

def augment_data(train):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1)
    ])
    train = train.map(lambda x, y: (data_augmentation(x, training=True), y))
    return train

def augment_data_multiple(train, num_repeats):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1)
    ])

    augmented_data = train
    for i in range(num_repeats):
        augmented_data = augmented_data.concatenate(
            train.map(lambda x, y: (data_augmentation(x, training=True), y))
        )

    return augmented_data
def train_model(model, train, val, logdir='logs'):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    model_checkpoint = ModelCheckpoint(os.path.join('checkpoints', model_name), monitor='val_loss',
                                       mode='min', verbose=1, save_best_only=True)

    hist = model.fit(train, epochs=25, validation_data=val, callbacks=[tensorboard_callback,
                                                                       early_stopping, lr_scheduler,model_checkpoint])
    return hist

def evaluate_model(model, test):
    pre = Precision()
    re = Recall()
    acc = CategoricalAccuracy()

    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        yhat = np.argmax(yhat, axis=1)

        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)

    print(f'Precision: {pre.result()}')
    print(f'Recall: {re.result()}')
    print(f'Categorical Accuracy: {acc.result()}')
def loss_diagram(hist):
    save_path = "plots/loss_plot_" + model_name + ".png"
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
    fig.savefig(save_path)

def accuracy_diagram(hist):
    save_path = "plots/accuracy_plot_" + model_name + ".png"
    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
    fig.savefig(save_path)

def train():
    remove_wrong_images(data_dir, image_exts)
    data = load_data(data_dir)
    train, val, test = split_data(data)
    train = augment_data(train)
    model = build_res_model()
    model.summary()
    hist = train_model(model, train, val)
    evaluate_model(model, test)
    save_trained_model(model, os.path.join('models', model_name))
    loss_diagram(hist)
    accuracy_diagram(hist)
    save_confusion_matrix(model,test)

def main():
    train()

if __name__ == '__main__':
    main()