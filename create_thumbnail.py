import cv2
import os
import re
import sys
from moviepy.editor import *
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import argparse
from os.path import isfile, join
import shutil
import imquality.brisque as brisque
import PIL.Image
import dlib

folder_path = "/global/D1/projects/soccer_clipping/events-Allsvenskan2019-minus15-pluss25/"
#folder_path = "/global/D1/projects/soccer_clipping/events-Eliteserien2019-minus15-pluss25/"
close_up_model = '/home/andrehus/egne_prosjekter/videoAndOutput/models/close_up_model.h5'
logo_detection_model = '/home/andrehus/egne_prosjekter/videoAndOutput/models/logo_detection/logo_detection.h5'
thumbnail_output = os.path.dirname(os.path.abspath(__file__)) + "/thumbnail_output/"
num_videos = 2
haarStr = "haar"
dlibStr = "dlib"
close_up_threshold = 0.6
brisque_threshold = 35
#Number of frames we want to extract
totalNumFrames = 50
#Number of frames to skip in the start
cutStartByFrames = 650
runFaceDetection = True
runBrisque = True
runLogoDetection = True
runCloseUpDetection = True


def main():
    parser = argparse.ArgumentParser(description="Thumbnail generator")
    parser.add_argument("destination", help="Destination of the input to be processed. Can be file or folder", nargs=1)
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument("-dlib", action='store_true', help="Dlib detection model is slower, but more presice.")
    group.add_argument("-haar", action='store_true', help="Haar detection model is faster, but less precise")
    #parser.add_argument("close_up threshold value")
    #parser.add_argument("brisque threshold value")
    #parser.add_argument("")

    args = parser.parse_args()
    print(args.destination)
    faceDetModel = ""

    if args.dlib:
        faceDetModel = dlibStr
        print("Using Dlib face detection model")
    else:
        faceDetModel = haarStr
        print("Using Haar face detection model")

    #print(args.accumulate(args.integers))
    #Flags that should be possible:

    #1. First argument should be file or folder to input
    #2. Flag which face detection model should be used
    #3. Decide the threshold values for close-up or brisque
    #4. Decide how many frames to process
    #5. Decide to exclude modules

    #Need to figure out how one specifies which models one wants to use when running
    
    try:
        if not os.path.exists(thumbnail_output):
            os.makedirs(thumbnail_output)

            for f in os.listdir(thumbnail_output):
                print(f)
                #os.remove(os.path.join(thumbnail_output, f))
    except OSError:
        print("Error: Couldnt create thumbnail_output directory")
        return

    global close_up_model
    global logo_detection_model
    close_up_model = keras.models.load_model(close_up_model)
    logo_detection_model = keras.models.load_model(logo_detection_model)
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
    totalFrames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
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
    frame_skip = (totalFrames-cutStartByFrames)//totalNumFrames
    numFramesExtracted = 0
    while(True):
        # reading from frame
        ret,frame = cam.read()
        if not ret:
            break
        if currentframe <= cutStartByFrames:
            currentframe += 1
            continue
        if currentframe % frame_skip == 0 and numFramesExtracted < totalNumFrames:
            # if video is still left continue creating images
            name = frames_folder + '/frames/frame' + str(currentframe) + '.jpg'
            width = int(frame.shape[1] * 0.5)
            height = int(frame.shape[0] * 0.5)
            dsize = (width, height)

            img = cv2.resize(frame, dsize) 
            cv2.imwrite(name, img)
            numFramesExtracted += 1
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
        
        if runBrisque:
            bestScore = 0
            for key in priority:
                score = predictBrisque(key)
                if finalThumbnail == "":
                    bestScore = score
                    finalThumbnail = key
                if score < brisque_threshold:
                    finalThumbnail = key
                    break
                if score < bestScore:
                    bestScore = score
                    finalThumbnail = key
        else:
            for key in priority:
                finalThumbnail = key
                break
        
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
    logos = []
    if runLogoDetection: 
        logo_probabilities = logo_detection_model.predict_generator(test_generator, TEST_SIZE)
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
            if runFaceDetection:
                face_size = detect_faces(image_path, faceDetModel)
                if face_size > 0:
                    print("Face detected: " + str(face_size) + "px")
                    priority_images[0][image_path] = probability
                else:
                    priority_images[1][image_path] = probability
            else:
                priority_images[1][image_path] = probability
        else:
            priority_images[2][image_path] = probability
            
    return priority_images

def predictBrisque(image_path):
    img = PIL.Image.open(image_path)
    brisqueScore = brisque.score(img)
    print("")
    print(image_path.split("/")[-1])
    print("Brisque score:")
    print(brisqueScore)
    return brisqueScore

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

def notRunningPrintFlags():
    print("")
    print("Specify with flag:")
    print("")
    print("Faster generating, but more unpresice:")
    print("python " + os.path.basename(__file__) + " -haar")
    print("")
    print("Slower generating, but more presice:")
    print("python " + os.path.basename(__file__) + " -dlib")

if __name__ == "__main__":
    main()
