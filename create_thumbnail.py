import cv2
import os
import re
import sys
from moviepy.editor import *
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

from os.path import isfile, join
import shutil
import imquality.brisque as brisque
import PIL.Image
import dlib

folder_path = "/global/D1/projects/soccer_clipping/events-Allsvenskan2019-minus15-pluss25/"
#folder_path = "/global/D1/projects/soccer_clipping/events-Eliteserien2019-minus15-pluss25/"
#folder_path = "/home/andrehus/egne_prosjekter/videoAndOutput/"
#model = keras.models.load_model('/home/andrehus/egne_prosjekter/videoAndOutput/models/thumbnail_vs_no_thumbnail.h5')
#model = keras.models.load_model('/home/andrehus/egne_prosjekter/videoAndOutput/models/thumbnail_vs_no_thumbnail_v2_model.h5')
close_up_model = keras.models.load_model('/home/andrehus/egne_prosjekter/videoAndOutput/models/close_up_model.h5')
logo_detection_model = keras.models.load_model('/home/andrehus/egne_prosjekter/videoAndOutput/models/logo_detection/logo_detection.h5')
thumbnail_output = os.path.dirname(os.path.abspath(__file__)) + "/thumbnail_output/"
num_videos = 5
haarStr = "haar"
dlibStr = "dlib"
close_up_threshold = 0.6
brisque_threshold = 35


def main():
    faceDetModel = ""
    try:
        argument = sys.argv[1]
        if argument == "-" + haarStr:
            faceDetModel = haarStr
            print("Using Haar face detection model")
        elif argument == "-" + dlibStr:
            faceDetModel = dlibStr
            print("Using Dlib face detection model")
        else:
            raise
    except:
        print("")
        print("Specify with flag:")
        print("")
        print("Faster generating, but more unpresice:")
        print("python " + os.path.basename(__file__) + " -haar")
        print("")
        print("Slower generating, but more presice:")
        print("python " + os.path.basename(__file__) + " -dlib")
        return
    
    try:
        if not os.path.exists(thumbnail_output):
            os.makedirs(thumbnail_output)

            for f in os.listdir(thumbnail_output):
                print(f)
                #os.remove(os.path.join(thumbnail_output, f))
    except OSError:
        print("Error: Couldnt create thumbnail_output directory")
        return
    i = 0
    for f in os.listdir(folder_path):
        if i >= num_videos:
            return
        name, ext = os.path.splitext(f)
        if ext == ".ts":
            create_thumbnail(name + ext, faceDetModel)
            i += 1
        

def create_thumbnail(video_filename, faceDetModel):
    print("Finding thumbnail for: ") 
    print(video_filename)
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
    # Don't start before 650 frames:
    startFrame = 650
    maxFrames = 3000
    # frames to skip
    frame_skip = 30
    while(currentframe < maxFrames):
        # reading from frame
        ret,frame = cam.read()
        if not ret:
            break
        if currentframe < startFrame-1:
            currentframe += 1
            continue
        if currentframe % frame_skip == 0:
            # if video is still left continue creating images
            name = frames_folder + '/frames/frame' + str(currentframe) + '.jpg'
            width = int(frame.shape[1] * 0.5)
            height = int(frame.shape[0] * 0.5)
            dsize = (width, height)

            img = cv2.resize(frame, dsize) 
            cv2.imwrite(name, img)
    
            # increasing counter so that it will
            # show how many frames are created
        currentframe += 1
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    priority_images = groupFrames(frames_folder, faceDetModel)
    finalThumbnail = ""
    for priority in priority_images:
        if finalThumbnail != "":
            break
        priority = dict(sorted(priority.items(), key=lambda item: item[1], reverse=True))
        print("")
        print("Frames in group: ")
        for key in priority:
            print(key.split("/")[-1])
        bestScore = 0
        for key in priority:
            score = predictBrisque(key)
            if finalThumbnail == "":
                bestScore = score
                finalThumbnail = key
            if score < brisque_threshold:
                image = key
                break
            if score < bestScore:
                bestScore = score
                finalThumbnail = key

        
    if finalThumbnail != "":
        newName = video_filename.split(".")[0] + "_thumbnail.jpg"
        imageName = finalThumbnail.split("/")[-1].split(".")[0]
        frameNum = int(imageName.replace("frame", ""))
        cam = cv2.VideoCapture(video_path)
        curFrame = 0
        while(True):
            ret, frame = cam.read()
            if curFrame == frameNum:
                cv2.imwrite(thumbnail_output + newName, frame)
                break
            curFrame += 1

        cam.release()
        cv2.destroyAllWindows()
        print("")
        print("Final thumbnail frame number: " + str(frameNum))
        try:
            shutil.rmtree(frames_folder)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        return
    return



def predictBrisque(image_path):
    img = PIL.Image.open(image_path)
    brisqueScore = brisque.score(img)
    print("")
    print(image_path.split("/")[-1])
    print("Brisque score:")
    print(brisqueScore)
    return brisqueScore

def groupFrames(frames_folder, faceDetModel):
    
    #frames_folder is your directory path as string
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
        if probability > 0.1:
            logos.append(image_path) 

    probabilities = close_up_model.predict_generator(test_generator, TEST_SIZE)
    priority_images = [{} for x in range(4)]

    for index, probability in enumerate(probabilities):
        print("")
        print(test_generator.filenames[index])
        print("close-up prediction score: " + str(probability))
        image_path = frames_folder + "/" + test_generator.filenames[index]
        if image_path in logos:
            print("Logo detected")
            priority_images[3][image_path] = probability
        elif probability > close_up_threshold:
            face_size = detect_faces(image_path, faceDetModel)
            if face_size > 0:
                print("Face detected: " + str(face_size) + "px")
                priority_images[0][image_path] = probability
            else:
                priority_images[1][image_path] = probability
        else:
            priority_images[2][image_path] = probability
            
    return priority_images



def detect_faces(image, faceDetModel):
    biggestFace = 0

    if faceDetModel == dlibStr:

        detector = dlib.get_frontal_face_detector()
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        for result in faces:
            x = result.left()
            y = result.top()
            x1 = result.right()
            y1 = result.bottom()
            size = y1-y
            if biggestFace < size:
                biggestFace = size

    elif faceDetModel == haarStr:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        biggestFace = 0
        for (x, y, w, h) in faces:
            if biggestFace < h:
                biggestFace = h

    else:
        print("No face detection model in use")

    return biggestFace

if __name__ == "__main__":
    main()
