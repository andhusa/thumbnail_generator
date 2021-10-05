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
import imquality.brisque as brisque
import PIL.Image

#folder_path = "/global/D1/projects/soccer_clipping/events-Allsvenskan2019-minus15-pluss25/"
#folder_path = "/global/D1/projects/soccer_clipping/events-Eliteserien2019-minus15-pluss25/"
folder_path = "/home/andrehus/egne_prosjekter/videoAndOutput/"
model = keras.models.load_model('/home/andrehus/egne_prosjekter/videoAndOutput/models/thumbnail_vs_no_thumbnail.h5')
logo_detection_model = keras.models.load_model('/home/andrehus/egne_prosjekter/videoAndOutput/models/logo_detection/logo_detection.h5')
thumbnail_output = folder_path + "/thumbnail_output/"
num_videos = 1

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
    # Don't start before 500 frames:
    currentframe = 500
    
    # frames to skip
    frame_skip = 25
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
        priority = dict(sorted(priority.items(), key=lambda item: item[1], reverse=True))
        print(priority)
        highestPri = 0
        image = ""
        for key in priority:
            print(key)
            if highestPri < priority[key]:
                score = predictBeauty(key)
                if score > 50:
                    highestPri = priority[key]
                    image = key
            
        if image != "":
            newName = video_filename.split(".")[0] + "_thumbnail.jpg"
            shutil.copy(image, thumbnail_output + newName)
            try:
                #shutil.rmtree(frames_folder)
                pass
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
            return



def predictBeauty(image_path):
    img = PIL.Image.open(image_path)
    beautyScore = brisque.score(img)
    print(image_path)
    print("Beauty score:")
    print(beautyScore)
    return beautyScore

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
    logo_probabilities = logo_detection_model.predict_generator(test_generator, TEST_SIZE)
    logos = []
    for index, probability in enumerate(logo_probabilities):
        image_path = frames_folder + "/" + test_generator.filenames[index]
        if probability > 0.5:
            print(image_path)
            print("LOGO")
            logos.append(image_path)

    probabilities = model.predict_generator(test_generator, TEST_SIZE)
    priority_images = [{} for x in range(5)]

    for index, probability in enumerate(probabilities):

        image_path = frames_folder + "/" + test_generator.filenames[index]
        #Beauty score 
        #print("")
        #print(""))
        if image_path in logos:
            continue
        print(image_path)
        print("probability:" + str(probability))
        if probability > 0.5:
            #print(image_path)
            #print("Probability: " + str(probability[0]*100) + " thumbnail")
            #img = PIL.Image.open(image_path)
            #beautyScore = brisque.score(img)
            #print(image_path)
            #print(beautyScore)
            print("Big face detected: " + str(detect_faces(image_path)))
            if not detect_faces(image_path):
                #os.remove(image_path)
                priority_images[1][image_path] = probability
            else:
                priority_images[0][image_path] = probability
        else:
            #print(image_path)
            #print("Probability: " + str((1-probability[0])*100) + " no-thumbnail")
            if not detect_faces(image_path):
                pass
                #os.remove(image_path)
            else:
                priority_images[2][image_path] = probability
            

    #print("priority_images:")
    #print(priority_images)
    return priority_images



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
        #Want the size of face to be bigger than 80 pixels in at least one dimension
        if w > 80 or h > 80:
            return True

    return False

if __name__ == "__main__":
    main()
