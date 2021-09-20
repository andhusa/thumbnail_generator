import cv2
import os
import re
from moviepy.editor import *
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import imutils
from os.path import isfile, join
import shutil

folder_path = "/global/D1/projects/soccer_clipping/events-Eliteserien2019-minus15-pluss25/"
#folder_path = "/home/andrehus/egne_prosjekter/videoAndOutput/"
model = keras.models.load_model('/home/andrehus/egne_prosjekter/videoAndOutput/models/thumbnail_vs_no_thumbnail.h5')
thumbnail_output = folder_path + "/thumbnail_output/"
num_videos = 50

def main():
    try:
        if not os.path.exists(thumbnail_output):
            os.makedirs(thumbnail_output)
    except OSError:
        print("Error: Couldnt create thumbnail_output directory")
    i = 0
    for f in os.listdir(folder_path):
        if i >= num_videos:
            return
        name, ext = os.path.splitext(f)
        if ext == ".ts":
            create_thumbnail(name + ext)
            i += 1
        

def create_thumbnail(video_filename):
    print("PREDICTING")
    video_path = folder_path + video_filename
    frames_folder = folder_path + video_filename.split(".")[0] + "_frames"
    # Read the video from specified path
    cam = cv2.VideoCapture(video_path)
    
    try:
        # creating a folder for frames

        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)
            os.makedirs(frames_folder + "/frames")
            print("created folder: " + frames_folder + "/frames")
    
    # if not created then raise error
    except OSError:
        print ('Error: Couldnt create directory')
    
    # frame
    currentframe = 0
    # frames to skip
    frame_skip = 60
    while(True):
        # reading from frame
        ret,frame = cam.read()
        if not ret:
            break
        if currentframe % frame_skip == 0:
            # if video is still left continue creating images
            name = frames_folder + '/frames/frame' + str(currentframe) + '.jpg'
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
        currentframe += 1
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    priority_images = predictAndRemove(frames_folder)
    for priority in priority_images:
        for image in priority:
            print(image)
            newName = video_filename.split(".")[0] + "_thumbnail.jpg"
            shutil.copy(image, thumbnail_output + newName)
            try:
                shutil.rmtree(frames_folder)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
            
            return
    #fileName = selectMean()
    #print(fileName)



def predictAndRemove(frames_folder):
    
    #dir is your directory path as string
    test_data_generator = ImageDataGenerator(rescale=1./255)
    IMAGE_SIZE = 200
    TEST_SIZE = len(next(os.walk(frames_folder + "/frames"))[2]) 
    print("TEST SIZE: " + str(TEST_SIZE))
    IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
    test_generator = test_data_generator.flow_from_directory(
        frames_folder,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=1,
        class_mode="binary", 
        shuffle=False)
    probabilities = model.predict_generator(test_generator, TEST_SIZE)
    images_with_faces = []
    priority_images = [[] for x in range(5)]
    i = 0
    for index, probability in enumerate(probabilities):
        numfiles = len(next(os.walk(frames_folder + "/frames"))[2])
        if( numfiles < 2):
            break
        image_path = frames_folder + "/" + test_generator.filenames[index]
        i += 1
        #print("")
        #print("")
        #print("PHOTO NUMBER " + str(i))
        
        
        if probability > 0.75:
            #print(image_path)
            #print("Probability: " + str(probability[0]*100) + " thumbnail")
            
            #print("Big face detected: " + str(detect_faces(image_path)))
            if not detect_faces(image_path):
                #os.remove(image_path)
                priority_images[2].append(image_path)
            else:
                priority_images[0].append(image_path)
                images_with_faces.append(image_path)
                return priority_images

        elif probability > 0.5:
            #print(image_path)
            #print("Not clear thumbnail")
            #print("Probability: " + str(probability[0]*100) + " thumbnail")
            #print("Big face detected: " + str(detect_faces(image_path)))
            if not detect_faces(image_path):
                #os.remove(image_path)
                priority_images[3].append(image_path)
            else:
                priority_images[1].append(image_path)
                images_with_faces.append(image_path)
        else:
            #print(image_path)
            #print("Probability: " + str((1-probability[0])*100) + " no-thumbnail")
            if not detect_faces(image_path):
                os.remove(image_path)
            else:
                images_with_faces.append(image_path)
                priority_images[4].append(image_path)
            
    #print("images with faces:")
    #for i in images_with_faces:
    #    print(i)

    #print("priority_images:")
    #print(priority_images)
    return priority_images


def selectMean():
    regex = re.compile(r'\d+')

    onlyfiles = [f for f in os.listdir(frames_folder + "/frames") if isfile(join(frames_folder + "/frames", f))]
    print(onlyfiles)
    frames = []
    for i in onlyfiles:
        frameNum = regex.findall(i)
        for i in frameNum:
            frames.append(int(frameNum[0]))

    if len(frames) == 0:
        raise Exception("No framenumber in the filenames of the frame folder")
    totalFrameNum = 0
    print(frames)
    for frame in frames:
        totalFrameNum += frame
    meanFrame = int(totalFrameNum / len(frames))
    print("meanframe: " + str(meanFrame))
    takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))
    closestToMean = takeClosest(meanFrame, frames)
    print(closestToMean)
    finalFile = ""
    for file in onlyfiles:
        if str(closestToMean) in file:
            finalFile = file

    return finalFile

def detect_faces(image):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Read the input image
    img = cv2.imread(image)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        if w > 80 or h > 80:
            return True

    return False

if __name__ == "__main__":
    main()
