import os
import re
import cv2
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
#import numpy as np
from os.path import isdir, isfile
import shutil
import imquality.brisque as brisque
import PIL.Image
import numpy as np

#All photos has to be in same folder and the name should tell the difference

#logo detection model:
#model = keras.models.load_model('/home/andrehus/egne_prosjekter/videoAndOutput/models/logo_detection/logo_detection.h5')

current_path = os.path.dirname(os.path.abspath(__file__))
#logo detection model
logo_detection = current_path + '/models/logo_detection.h5'
logo_detection = '/home/andrehus/image_classifier/logos_soccernet_reduced.h5'
#close up model:
close_up = current_path + '/models/close_up_model.h5'
close_up = '/home/andrehus/image_classifier/close_up_model_3_downscaled.h5'
#close_up = '/home/andrehus/image_classifier/close_up_model_3.h5'

#model = keras.models.load_model(close_up)
model = keras.models.load_model(logo_detection)




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
        #probability = 1 - probability
        image_path = folder + "/" + test_generator.filenames[index]
        fileName = test_generator.filenames[index].split("/")[-1]
        className = fileName.split("_")[0]
        print(className)
        print(probability)
        if probability > probabilityThr:
            if className == "closeUp" or className == "Logo":
                trueP += 1
                #print("trueP")
            else:
                falseP += 1
                #print("falseP")
        else:
            if className == "closeUp" or className == "Logo":
                falseN += 1
                #print("falseN")
            else:
                trueN += 1
                #print("trueN")

    
    print("True positives: " + str(trueP))
    print("False positives: " + str(falseP))
    print("True negatives: " + str(trueN))
    print("False negatives: " + str(falseN))
    numCases = trueP + falseP + trueN + falseN
    print("Num cases: " + str(numCases))
    if (trueP + falseP == 0) or (trueP + falseN == 0):
        print("Can't divide by zero")
        return
    precision = trueP / (trueP + falseP)
    recall = trueP / (trueP + falseN)
    accuracy = (trueP + trueN) / numCases
    print("Probability threshold: " + str(probabilityThr))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Accuracy: " + str(accuracy))


def testImageQualityPredictor(fileName):
    if os.path.isfile(fileName):
        cutStartByFrames = 650
        totalFramesToExtract = 65
        downscaleOnProcessing = 0.625
        video_filename = fileName.split('/')[-1]
        frames_folder_outer = os.path.dirname(os.path.abspath(__file__)) + "/extractedFrames/"
        frames_folder = frames_folder_outer + video_filename.split(".")[0] + "_frames/"
        fullsized = frames_folder + "fullsized/"
        downscaled = frames_folder + "downscaled/"
        cam = cv2.VideoCapture(fileName)
        totalFrames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        if totalFrames < cutStartByFrames:
            print("The starting cut frame doesn't exist")
            return
        try:
            # creating a folder for frames
            if not os.path.exists(frames_folder):
                os.makedirs(frames_folder)
                os.makedirs(fullsized)
                os.makedirs(downscaled)
        except OSError:
            print ('Error: Couldnt create directory')
        

        fps = cam.get(cv2.CAP_PROP_FPS)
        duration = totalFrames/fps
        currentframe = 0
        # frames to skip
        frame_skip = (totalFrames-cutStartByFrames)//totalFramesToExtract
        numFramesExtracted = 0
        true = 0
        false = 0
        width = 0
        height = 0 
        while(True):
            # reading from frame
            ret,frame = cam.read()
            if not ret:
                break
            if currentframe <= cutStartByFrames:
                currentframe += 1
                continue
            if currentframe % frame_skip == 0 and numFramesExtracted < totalFramesToExtract:
                # if video is still left continue creating images
                fullsizedName = fullsized + 'frame' + str(currentframe) + '.jpg'
                downscaledName = downscaled + 'frame' + str(currentframe) + '.jpg'
                width = int(frame.shape[1] * downscaleOnProcessing)
                height = int(frame.shape[0] * downscaleOnProcessing)
                dsize = (width, height)

                cv2.imwrite(fullsizedName, frame)
                img = cv2.resize(frame, dsize)
                cv2.imwrite(downscaledName, img)

                numFramesExtracted += 1
                # increasing counter so that it will
                # show how many frames are created
            currentframe += 1
        
        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()
        results = {}
        for f in os.listdir(fullsized):
            score = predictBrisque(fullsized + f)
            name = f.split('.')[0]
            results[name] = [score]

        for f in os.listdir(downscaled):
            score = predictBrisque(downscaled + f)
            name = f.split('.')[0]
            results[name].append(score)
            
        true = 0
        false = 0

        for r in results:
            if results[r][0] > results[r][1]:
                true += 1
            else:
                false += 1

        accuracy = true / (true + false)
        
        print(str(width) + " x " + str(height))
        print("Accuracy: " + str(accuracy))

        printAvgFileSizeFolder(fullsized)
        printAvgFileSizeFolder(downscaled)


def predictBrisque(image_path):
    img = cv2.imread(image_path)
    print(img.shape)
    brisqueScore = brisque.score(img)
    #brisqueScore = 1
    print("")
    print(image_path.split("/")[-1])
    print("Brisque score:")
    print(brisqueScore)
    return brisqueScore

def printAvgFileSizeFolder(folder):
    sizes = []
    for f in os.listdir(folder):
        sizes.append(os.path.getsize(folder + f))
    avgSize = sum(sizes) / len(sizes)
    print("Avg file size," + folder.split('/')[-1] +" is: " + str(avgSize))

def test_blur(positive_folder, negativeFolder):
    max_cases_per_folder = 1000
    positive_scores = []
    negative_scores = []
    blur_threshold = 0.7
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    i = 0
    for filename in os.listdir(positive_folder):
        if i > max_cases_per_folder:
            break
        name, ext = os.path.splitext(filename)
        if ext == ".jpg" or ext == ".png":
            i += 1
            score = get_blur_degree(positive_folder + filename)
            print(score)
            if score > blur_threshold:
                tp += 1
            else:
                fn += 1
            positive_scores.append(score)
    i = 0
    for filename in os.listdir(negativeFolder):
        if i > max_cases_per_folder:
            break
        name, ext = os.path.splitext(filename)
        if ext == ".jpg" or ext == ".png":
            i += 1
            score = get_blur_degree(negativeFolder + filename)
            print(score)
            if score > blur_threshold:
                fp += 1
            else:
                tn += 1
            negative_scores.append(score)
    positive_scores_mean = sum(positive_scores) / len(positive_scores)
    negative_scores_mean = sum(negative_scores) / len(negative_scores)
    print("Positive scores mean: " + str(positive_scores_mean))
    print("Negative scores mean: " + str(negative_scores_mean))
    print("True positives: " + str(tp))
    print("False positives: " + str(fp))
    print("True negatives: " + str(tn))
    print("False negatives: " + str(fn))
    numCases = tp + fp + tn + fn
    print("Num cases: " + str(numCases))
    if (tp + fp == 0) or (tp + fn == 0):
        print("Can't divide by zero")
        return
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / numCases
    print("Probability threshold: " + str(blur_threshold))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Accuracy: " + str(accuracy))

def get_blur_degree(image_file, sv_num=10):
    img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    u, s, v = np.linalg.svd(img)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv/total_sv

if __name__ == "__main__":

    folder = "/global/D1/projects/soccer_clipping/closeUpSetDownscaledStructured/test"
    folder = "/global/D1/projects/soccer_clipping/AllsvenskanTestLogoStructured/test"
    main(folder)

    folder = "/global/D1/projects/soccer_clipping/events-Eliteserien2019-minus15-pluss25/"
    positive_folder = "/global/D1/projects/soccer_clipping/Eliteserien/Test/Logo/"
    negative_folder = "/global/D1/projects/soccer_clipping/Eliteserien/Test/Background/"
    positive_folder = "/global/D1/projects/soccer_clipping/SoccerNetLogos/Test/Background/"
    negative_folder = "/global/D1/projects/soccer_clipping/SoccerNetLogos/Test/Logo/"
    #test_blur(positive_folder, negative_folder)
    #Testing Brisque:

    #for f in os.listdir(folder):
    #    if f.split('.')[-1] == 'ts':
    #        testImageQualityPredictor(folder + f)
    #        break

