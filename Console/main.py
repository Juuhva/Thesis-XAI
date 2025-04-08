import os
from collections import Counter

import cv2
import keras
import numpy as np
import pandas as pd
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.saving import load_model
import tensorflow as tf
from matplotlib import pyplot as plt
from IPython.display import Image, display
from sklearn.metrics import classification_report

def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )


    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def load_trained_model(model_path):
    new_model = load_model(model_path)
    return new_model

def make_prediction(model, img_path):
    img = cv2.imread(img_path)
    resize = tf.image.resize(img, (256, 256))
    yhat = model.predict(np.expand_dims(resize / 255, 0))

    predicted_class = np.argmax(yhat)
    confidence = np.max(yhat) * 100

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    classes = ["cat", "dog","human"]

    if predicted_class < len(classes):
        print(f"Predicted class: {classes[predicted_class]}")
    else:
        print("Not recognized")

    print(f"Confidence: {confidence:.2f}%")

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = plt.colormaps["jet"]

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)

    display(Image(cam_path))

def run():
    loaded_img = "../test_imgs/cat/cat1.jpg"
    loaded_model = load_trained_model(os.path.join("..",'checkpoints', model_name))
    preprocess_input = keras.applications.xception.preprocess_input
    img_size = (256, 256)
    img_array = preprocess_input(get_img_array(loaded_img, size=img_size))
    loaded_model.summary()
    make_prediction(loaded_model, loaded_img)
    heatmap = make_gradcam_heatmap(img_array, loaded_model, "conv2d_11")
    plt.matshow(heatmap)
    plt.show()
    save_and_display_gradcam(loaded_img, heatmap)
    run_model_test("../checkpoints/MultipleAug.keras","../test_imgs")

def run_model_test(model_path, test_dir, target_size=(256, 256), batch_size=32):
    model = load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    true_labels = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    predictions = model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=2
    )

    print("Model metrics:")
    print(report)

    return report

model_name = 'MultipleAug.keras'

def main():
    run()

if __name__ == '__main__':
    main()