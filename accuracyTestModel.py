import os
import re
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from os.path import isdir
import shutil

#All photos has to be in same folder and the name should tell the difference

#logo detection model:
#model = keras.models.load_model('/home/andrehus/egne_prosjekter/videoAndOutput/models/logo_detection/logo_detection.h5')

#close up model:
model = keras.models.load_model('/home/andrehus/egne_prosjekter/videoAndOutput/models/close_up_model.h5')

probabilityThr = 0.5

def main(folder):
    test_data_generator = ImageDataGenerator(rescale=1./255)
    IMAGE_SIZE = 200
    TEST_SIZE = len(next(os.walk(folder + "/testClass"))[2])
    print("TEST SIZE: " + str(TEST_SIZE))
    IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
    test_generator = test_data_generator.flow_from_directory(
        folder,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=1,
        class_mode="binary",
        shuffle=False)
    probabilities = model.predict_generator(test_generator, TEST_SIZE)
    trueP = 0
    falseP = 0
    trueN = 0
    falseN = 0
    for index, probability in enumerate(probabilities):
        image_path = folder + "/" + test_generator.filenames[index]
        fileName = test_generator.filenames[index].split("/")[-1]
        className = fileName.split("_")[0]
        print(className)
        print(probability)
        if probability > probabilityThr:
            if className == "closeUp" or className == "logo":
                trueP += 1
                print("trueP")
            else:
                falseP += 1
                print("flaseP")
        else:
            if className == "closeUp" or className == "logo":
                falseN += 1
                print("falseN")
            else:
                trueN += 1
                print("trueN")

    print("True positives: " + str(trueP))
    print("False positives: " + str(falseP))
    print("True negatives: " + str(trueN))
    print("False negatives: " + str(falseN))
    if (trueP + falseP == 0) or (trueP + falseN == 0):
        print("Can't divide by zero")
        return
    precision = trueP / (trueP + falseP)
    recall = trueP / (trueP + falseN)
    accuracy = (trueP + trueN) / (trueP + trueN + falseP + trueN)
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Accuracy: " + str(accuracy))

if __name__ == "__main__":
    #folder = "/home/andrehus/photos/outputStructured/test"
    #folder = "/home/andrehus/photos/thumbnail-dataset/test"
    folder = "/global/D1/projects/soccer_clipping/closeUpSetStructured/test"
    main(folder)

