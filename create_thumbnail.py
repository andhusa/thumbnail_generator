import cv2
import os
import re
import sys
from moviepy.editor import *
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import argparse
from os.path import isfile, join, isdir
import shutil
import imquality.brisque as brisque
import PIL.Image
import dlib

folder_path = "/global/D1/projects/soccer_clipping/events-Allsvenskan2019-minus15-pluss25/"
#folder_path = "/global/D1/projects/soccer_clipping/events-Eliteserien2019-minus15-pluss25/"

thumbnail_output = os.path.dirname(os.path.abspath(__file__)) + "/thumbnail_output/"

num_videos = 2
haarStr = "haar"
dlibStr = "dlib"

runCloseUpDetection = True


def main(close_up_model, logo_detection_model):
    #Default values
    close_up_threshold = 0.6
    brisque_threshold = 35
    totalNumFrames = 50
    cutStartByFrames = 650

    parser = argparse.ArgumentParser(description="Thumbnail generator")
    parser.add_argument("destination", nargs=1, help="Destination of the input to be processed. Can be file or folder")
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument("-dlib", action='store_true', help="Dlib detection model is slower, but more presice.")
    group.add_argument("-haar", action='store_true', help="Haar detection model is faster, but less precise")
    
    #Flags that excludes models running
    parser.add_argument("-xf", "--xFaceDetection", default=True, action="store_false", help="Don't run the face detection")
    parser.add_argument("-xb", "--xBrisque", default=True, action="store_false", help="Don't run Brisque")
    parser.add_argument("-xl", "--xLogoDetection", default=True, action="store_false", help="Don't run logo detection")

    #Flags fixing default values
    parser.add_argument("-cuthr", "--closeUpThreshold", type=restricted_float, default=[close_up_threshold], nargs=1, help="The threshold value for the close-up detection model. The value must be between 0 and 1. The default is: " + str(close_up_threshold))
    parser.add_argument("-brthr", "--brisqueThreshold", type=float, default=[brisque_threshold], nargs=1, help="The threshold value for the brisque model. The default is: " + str(brisque_threshold))
    parser.add_argument("-csfr", "--cutStartFrames", type=positive_int, default=[cutStartByFrames], nargs=1, help="The number of frames to cut from start of the video. These will not be processed in the thumbnail selection. The default is: " + str(cutStartByFrames))
    parser.add_argument("-nf", "--numberOfFrames", type=above_zero_int, default=[totalNumFrames], nargs=1, help="Number of frames to be extracted from the video for the thumbnail selection process. The default is: " + str(totalNumFrames))

    args = parser.parse_args()
    destination = args.destination[0]
    runFaceDetection = args.xFaceDetection
    runBrisque = args.xBrisque
    runLogoDetection = args.xLogoDetection
    close_up_threshold = args.closeUpThreshold[0]
    brisque_threshold = args.brisqueThreshold[0]
    cutStartByFrames = args.cutStartFrames[0]
    totalNumFrames = args.numberOfFrames[0]

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
    processFolder = False
    processFile = False
    if os.path.isdir(destination):
        processFolder = True
        if destination[-1] != "/":
            destination = destination + "/"
        print("isfolder")
    elif os.path.isfile(destination):
        processFile = True
        print("isfile")
    else:
        print("Error: The input destination was neither file or directory") 
        return

    try:
        if not os.path.exists(thumbnail_output):
            os.makedirs(thumbnail_output)

            for f in os.listdir(thumbnail_output):
                print(f)
                #os.remove(os.path.join(thumbnail_output, f))
    except OSError:
        print("Error: Couldn't create thumbnail_output directory")
        return


    close_up_model = keras.models.load_model(close_up_model)
    if runLogoDetection:
        logo_detection_model = keras.models.load_model(logo_detection_model)

    if processFile:
        name, ext = os.path.splitext(destination)
        if ext == ".ts":
            create_thumbnail(name + ext, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runBrisque, runLogoDetection, close_up_threshold, brisque_threshold, cutStartByFrames, totalNumFrames)
    elif processFolder:

        i = 0
        for f in os.listdir(destination):
            if i >= num_videos:
                return
            name, ext = os.path.splitext(f)
            if ext == ".ts":
                create_thumbnail(destination + name + ext, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runBrisque, runLogoDetection, close_up_threshold, brisque_threshold, cutStartByFrames, totalNumFrames)
                i += 1
        

def create_thumbnail(video_path, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runBrisque, runLogoDetection, close_up_threshold, brisque_threshold, cutStartByFrames, totalNumFrames):
    print("Finding thumbnail for: ")
    video_filename = video_path.split("/")[-1]
    frames_folder_outer = os.path.dirname(os.path.abspath(__file__)) + "/extractedFrames/"
    frames_folder = frames_folder_outer + video_filename.split(".")[0] + "_frames"
    # Read the video from specified path
    cam = cv2.VideoCapture(video_path)
    totalFrames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    if totalFrames < cutStartByFrames:
        print("The starting cut frame doesn't exist")
        return
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
    priority_images = groupFrames(frames_folder, close_up_model, logo_detection_model ,faceDetModel, runFaceDetection, runBrisque, runLogoDetection, close_up_threshold, brisque_threshold)
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
            shutil.rmtree(frames_folder_outer)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        return
    return

def groupFrames(frames_folder, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runBrisque, runLogoDetection, close_up_threshold, brisque_threshold):
    
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

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def positive_int(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an int literal" % (x,))
    if x < 0:
        raise argparse.ArgumentTypeError("%r not a positive int"%(x,))
    return x

def above_zero_int(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an int literal" % (x,))
    if x <= 0:
        raise argparse.ArgumentTypeError("%r not above zero"%(x,))
    return x

if __name__ == "__main__":
    close_up_model = '/home/andrehus/egne_prosjekter/videoAndOutput/models/close_up_model.h5'
    logo_detection_model = '/home/andrehus/egne_prosjekter/videoAndOutput/models/logo_detection/logo_detection.h5'
    main(close_up_model, logo_detection_model)
