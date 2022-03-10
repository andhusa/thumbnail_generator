import cv2
import os
import math
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import argparse
from os.path import isfile, isdir
import shutil
import imquality.brisque as brisque
import dlib
import time
from mtcnn.mtcnn import MTCNN

folder_path = "/global/D1/projects/soccer_clipping/events-Allsvenskan2019-minus15-pluss25/"
#video="/global/D1/projects/soccer_clipping/events-Allsvenskan2019-minus15-pluss25/akwaxywqi4qo3.ts"
#folder_path = "/global/D1/projects/soccer_clipping/events-Eliteserien2019-minus15-pluss25/"

current_path = os.path.dirname(os.path.abspath(__file__))

haarXml = current_path + '/models/haarcascade_frontalface_default.xml'
modelFile = current_path + "/models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = current_path + "/models/deploy.prototxt.txt"
thumbnail_output = current_path + "/thumbnail_output/"
excluded_images = current_path + "/excluded_images/"
surmaLogoModel = current_path + '/models/logo_detection.h5'
surmaCloseupModel = current_path + '/models/close_up_model.h5'

haarStr = "haar"
dlibStr = "dlib"
mtcnnStr = "mtcnn"
dnnStr = "dnn"
surmaStr = "surma"

#The probability score the image classifying model gives, is depending on which class it is basing the score on.
#It could be switched
close_up_model_inverted = False

def main():
    #Default values
    close_up_threshold = 0.75
    brisque_threshold = 35
    totalFramesToExtract = 50
    faceDetModel = dlibStr
    framerateExtract = None
    fpsExtract = None
    cutStartSeconds = 0
    cutEndSeconds = 0
    downscaleOnProcessing = 0.5
    downscaleOutput = 1.0
    annotationSecond = None
    beforeAnnotationSecondsCut = None
    afterAnnotationSecondsCut = None
    staticThumbnailSec = None
    logo_model_name = surmaStr
    logo_detection_model = ""
    close_up_model_name = surmaStr
    close_up_model = ""

    parser = argparse.ArgumentParser(description="Thumbnail generator")
    parser.add_argument("destination", nargs=1, help="Destination of the input to be processed. Can be file or folder")

    #Logo detection models
    logoGroup = parser.add_mutually_exclusive_group(required=False)
    logoGroup.add_argument("-Lsurma", action='store_true', help="Surma model used for logo detection.")
    logoGroup.add_argument("-xl", "--xLogoDetection", default=True, action="store_false", help="Don't run logo detection")

    #Close-up detection models
    closeupGroup = parser.add_mutually_exclusive_group(required=False)
    closeupGroup.add_argument("-Csurma", action='store_true', help="Surma model used for close-up detection.")
    closeupGroup.add_argument("-xc", "--xCloseupDetection", default=True, action="store_false", help="Don't run close-up detection")

    #IQA models
    iqaGroup = parser.add_mutually_exclusive_group(required=False)
    iqaGroup.add_argument("-IQAocampo", action='store_true', help="Ocampo model used for image quality assessment.")
    iqaGroup.add_argument("-xi", "--xIQA", default=True, action="store_false", help="Don't run image quality prediction")


    #Face models
    faceGroup = parser.add_mutually_exclusive_group(required = False)
    faceGroup.add_argument("-dlib", action='store_true', help="Dlib detection model is slow, but presice.")
    faceGroup.add_argument("-haar", action='store_true', help="Haar detection model is fast, but unprecise.")
    faceGroup.add_argument("-mtcnn", action='store_true', help="MTCNN detection model is slow, but precise.")
    faceGroup.add_argument("-dnn", action='store_true', help="DNN detection model is fast and precise.")

    #Flags that excludes models running
    faceGroup.add_argument("-xf", "--xFaceDetection", default=True, action="store_false", help="Don't run the face detection")

    #Flags fixing default values
    parser.add_argument("-cuthr", "--closeUpThreshold", type=restricted_float, default=[close_up_threshold], nargs=1, help="The threshold value for the close-up detection model. The value must be between 0 and 1. The default is: " + str(close_up_threshold))
    parser.add_argument("-brthr", "--brisqueThreshold", type=float, default=[brisque_threshold], nargs=1, help="The threshold value for the image quality predictor model. The default is: " + str(brisque_threshold))
    parser.add_argument("-css", "--cutStartSeconds", type=positive_int, default=[cutStartSeconds], nargs=1, help="The number of seconds to cut from start of the video. These seconds of video will not be processed in the thumbnail selection. The default value is: " + str(cutStartSeconds))
    parser.add_argument("-ces", "--cutEndSeconds", type=positive_int, default=[cutEndSeconds], nargs=1, help="The number of seconds to cut from the end of the video. These seconds of video will not be processed in the thumbnail selection. The default value is: " + str(cutEndSeconds))
    numFrameExtractGroup = parser.add_mutually_exclusive_group(required = False)
    numFrameExtractGroup.add_argument("-nfe", "--numberOfFramesToExtract", type=above_zero_int, default=[totalFramesToExtract], nargs=1, help="Number of frames to be extracted from the video for the thumbnail selection process. The default is: " + str(totalFramesToExtract))
    numFrameExtractGroup.add_argument("-fre", "--framerateToExtract", type=restricted_float, default=[framerateExtract], nargs=1, help="The framerate wanted to be extracted from the video for the thumbnail selection process.")
    numFrameExtractGroup.add_argument("-fpse", "--fpsExtract", type=above_zero_float, default=[fpsExtract], nargs=1, help="Number of frames per second to extract from the video for the thumbnail selection process.")
    parser.add_argument("-ds", "--downscaleProcessingImages", type=restricted_float, default=[downscaleOnProcessing], nargs=1, help="The value deciding how much the images to be processed should be downscaled. The default value is: " + str(downscaleOnProcessing))
    parser.add_argument("-dso", "--downscaleOutputImage", type=restricted_float, default=[downscaleOutput], nargs=1, help="The value deciding how much the output thumbnail image should be downscaled. The default value is: " + str(downscaleOutput))
    parser.add_argument("-as", "--annotationSecond", type=positive_int, default=[annotationSecond], nargs=1, help="The second the event is annotated to in the video.")
    parser.add_argument("-bac", "--beforeAnnotationSecondsCut", type=positive_int, default=[beforeAnnotationSecondsCut], nargs=1, help="Seconds before the annotation to cut the frame extraction.")
    parser.add_argument("-aac", "--afterAnnotationSecondsCut", type=positive_int, default=[afterAnnotationSecondsCut], nargs=1, help="Seconds after the annotation to cut the frame extraction.")
    parser.add_argument("-st", "--staticThumbnailSec", type=positive_int, default=[staticThumbnailSec], nargs=1, help="To generate a static thumbnail from the video, this flag is used. The second the frame should be clipped from should follow as an argument. Running this flag ignores all the other flags.")


    args = parser.parse_args()
    destination = args.destination[0]
    staticThumbnailSec = args.staticThumbnailSec[0]
    runFaceDetection = args.xFaceDetection
    runIQA = args.xIQA
    runLogoDetection = args.xLogoDetection
    if not runLogoDetection:
        logo_model_name = ""
    runCloseUpDetection = args.xCloseupDetection
    if not runCloseUpDetection:
        close_up_model_name = ""

    close_up_threshold = args.closeUpThreshold[0]
    brisque_threshold = args.brisqueThreshold[0]
    cutStartSeconds = args.cutStartSeconds[0]
    cutEndSeconds = args.cutEndSeconds[0]
    totalFramesToExtract = args.numberOfFramesToExtract[0]
    framerateExtract = args.framerateToExtract[0]
    fpsExtract = args.fpsExtract[0]
    if fpsExtract:
        totalFramesToExtract = None
        framerateExtract = None
    if framerateExtract:
        totalFramesToExtract = None
        fpsExtract = None
    if totalFramesToExtract:
        framerateExtract = None
        fpsExtract = None
    downscaleOnProcessing = args.downscaleProcessingImages[0]
    downscaleOutput = args.downscaleOutputImage[0]
    annotationSecond = args.annotationSecond[0]
    beforeAnnotationSecondsCut = args.beforeAnnotationSecondsCut[0]
    afterAnnotationSecondsCut = args.afterAnnotationSecondsCut[0]

    if args.dlib:
        faceDetModel = dlibStr
        print("Using Dlib face detection model.")
    elif args.haar:
        faceDetModel = haarStr
        print("Using Haar face detection model.")
    elif args.mtcnn:
        faceDetModel = mtcnnStr
        print("Using MTCNN face detection model.")
    elif args.dnn:
        faceDetModel = dnnStr
        print("Using DNN face detection model.")

    processFolder = False
    processFile = False
    if os.path.isdir(destination):
        processFolder = True
        if destination[-1] != "/":
            destination = destination + "/"
        print("is folder")
    elif os.path.isfile(destination):
        processFile = True
        print("is file")
    else:
        print("Error: The input destination was neither file or directory")
        return

    try:
        if not os.path.exists(thumbnail_output):
            os.mkdir(thumbnail_output)

    except OSError:
        print("Error: Couldn't create thumbnail output directory")
        return

    if staticThumbnailSec:
        get_static(destination, staticThumbnailSec, downscaleOutput, thumbnail_output)
        return

    if close_up_model_name == surmaStr:
        close_up_model = keras.models.load_model(surmaCloseupModel)

    if logo_model_name == surmaStr:
        logo_detection_model = keras.models.load_model(surmaLogoModel)

    if processFile:
        name, ext = os.path.splitext(destination)
        if ext == ".ts" or ext == ".mp4":
            create_thumbnail(name + ext, downscaleOutput, downscaleOnProcessing, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runIQA, runLogoDetection, runCloseUpDetection, close_up_threshold, brisque_threshold, cutStartSeconds, cutEndSeconds, totalFramesToExtract, fpsExtract, framerateExtract, annotationSecond, beforeAnnotationSecondsCut, afterAnnotationSecondsCut)
    elif processFolder:

        for f in os.listdir(destination):
            name, ext = os.path.splitext(f)
            print(name + ext)
            if ext == ".ts" or ext == ".mp4":
                create_thumbnail(destination + name + ext,downscaleOutput , downscaleOnProcessing, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runIQA, runLogoDetection, runCloseUpDetection, close_up_threshold, brisque_threshold, cutStartSeconds, cutEndSeconds, totalFramesToExtract, fpsExtract, framerateExtract, annotationSecond, beforeAnnotationSecondsCut, afterAnnotationSecondsCut)




def create_thumbnail(video_path, downscaleOutput, downscaleOnProcessing, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runIQA, runLogoDetection, runCloseUpDetection, close_up_threshold, brisque_threshold, cutStartSeconds, cutEndSeconds, totalFramesToExtract, fpsExtract, framerateExtract, annotationSecond, beforeAnnotationSecondsCut, afterAnnotationSecondsCut):
    video_filename = video_path.split("/")[-1]
    frames_folder_outer = os.path.dirname(os.path.abspath(__file__)) + "/extractedFrames/"
    frames_folder = frames_folder_outer + video_filename.split(".")[0] + "_frames"
    # Read the video from specified path

    cam = cv2.VideoCapture(video_path)
    totalFrames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cam.get(cv2.CAP_PROP_FPS)

    duration = totalFrames/fps

    if annotationSecond:
        if beforeAnnotationSecondsCut:
            cutStartSeconds = annotationSecond - beforeAnnotationSecondsCut
        if afterAnnotationSecondsCut:
            cutEndSeconds = duration - (annotationSecond + afterAnnotationSecondsCut)


    cutStartFrames = fps * cutStartSeconds
    cutEndFrames = fps * cutEndSeconds


    if totalFrames < cutStartFrames + cutEndFrames:
        print("All the frames are cut out")
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

    remainingFrames = totalFrames - (cutStartFrames + cutEndFrames)
    remainingSeconds = remainingFrames / fps

    if fpsExtract:
        totalFramesToExtract = math.floor(remainingSeconds * fpsExtract)
    if framerateExtract:
        totalFramesToExtract = math.floor(remainingFrames * framerateExtract)


    currentframe = 0
    # frames to skip
    frame_skip = (totalFrames-(cutStartFrames + cutEndFrames))//totalFramesToExtract
    numFramesExtracted = 0
    stopFrame = totalFrames-cutEndFrames
    while(True):
        # reading from frame
        ret,frame = cam.read()
        if not ret:
            break
        if currentframe > stopFrame:
            break
        if currentframe <= cutStartFrames:
            currentframe += 1
            continue
        if currentframe % frame_skip == 0 and numFramesExtracted < totalFramesToExtract:
            # if video is still left continue creating images
            name = frames_folder + '/frames/frame' + str(currentframe) + '.jpg'
            width = int(frame.shape[1] * downscaleOnProcessing)
            height = int(frame.shape[0] * downscaleOnProcessing)
            dsize = (width, height)
            img = cv2.resize(frame, dsize)

            cv2.imwrite(name, img)
            numFramesExtracted += 1

        currentframe += 1

    priority_images = groupFrames(frames_folder, close_up_model, logo_detection_model ,faceDetModel, runFaceDetection, runIQA, runLogoDetection, runCloseUpDetection, close_up_threshold, brisque_threshold)
    finalThumbnail = ""
    excluded = []

    for priority in priority_images:
        if finalThumbnail != "":
            break
        priority = dict(sorted(priority.items(), key=lambda item: item[1], reverse=True))

        '''
        Blur detection not added:

        bestScore = 0
        blur_threshold = 0.6
        for key in priority:
            score = get_blur_degree(key)

            if finalThumbnail == "":
                bestScore = score
                finalThumbnail = key

            if score < blur_threshold:
                #finalThumbnail = key
                brisqueScore = predictBrisque(key)
                if brisqueScore < brisque_threshold:
                    finalThumbnail = key
                    break

            if score < bestScore:
                bestScore = score
                finalThumbnail = key

        '''

        if runIQA:
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
        #cam = cv2.VideoCapture(video_path)

        cam.set(1, frameNum)
        ret, frame = cam.read()
        if downscaleOutput != 1.0:
            width = int(frame.shape[1] * downscaleOutput)
            height = int(frame.shape[0] * downscaleOutput)
            dsize = (width, height)
            frame = cv2.resize(frame, dsize)

        cv2.imwrite(thumbnail_output + newName, frame)

        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()

        #secInVid = (frameNum / totalFrames) * duration

        try:
            shutil.rmtree(frames_folder_outer)

        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        return
    return

def groupFrames(frames_folder, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runIQA, runLogoDetection, runCloseUpDetection, close_up_threshold, brisque_threshold):
    test_generator = None
    TEST_SIZE = 0
    if runCloseUpDetection or runLogoDetection:
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

    priority_images = [{} for x in range(4)]
    if runCloseUpDetection:
        probabilities = close_up_model.predict_generator(test_generator, TEST_SIZE)

        for index, probability in enumerate(probabilities):

            #The probability score is inverted:
            if close_up_model_inverted:
                probability = 1 - probability

                image_path = frames_folder + "/" + test_generator.filenames[index]

                if image_path in logos:
                    priority_images[3][image_path] = probability

            elif probability > close_up_threshold:
                if runFaceDetection:
                    face_size = detect_faces(image_path, faceDetModel)
                    if face_size > 0:
                        priority_images[0][image_path] = face_size
                    else:
                        priority_images[1][image_path] = probability
                else:
                    priority_images[1][image_path] = probability
            else:
                priority_images[2][image_path] = probability
    else:
        probability = 1
        frames_folder = frames_folder + "/frames/"
        for image in os.listdir(frames_folder):
            image_path = frames_folder + image
            if image_path in logos:
                priority_images[3][image_path] = probability
            if runFaceDetection:
                face_size = detect_faces(image_path, faceDetModel)
                if face_size > 0:
                    priority_images[0][image_path] = face_size
                else:
                    priority_images[1][image_path] = probability
            else:
                priority_images[1][image_path] = probability
    return priority_images

def get_static(video_path, secondExtract, downscaleOutput, outputFolder):
    video_filename = video_path.split("/")[-1]
    frames_folder_outer = os.path.dirname(os.path.abspath(__file__)) + "/extractedFrames/"
    frames_folder = frames_folder_outer + video_filename.split(".")[0] + "_frames"

    cam = cv2.VideoCapture(video_path)
    totalFrames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cam.get(cv2.CAP_PROP_FPS)

    duration = totalFrames/fps


    cutStartFrames = fps * secondExtract


    if totalFrames < cutStartFrames:
        print("All the frames are cut out")
        return

    currentframe = 0
    while(True):
        # reading from frame
        ret,frame = cam.read()
        if not ret:
            break
        if currentframe <= cutStartFrames:
            currentframe += 1
            continue
        width = int(frame.shape[1] * downscaleOutput)
        height = int(frame.shape[0] * downscaleOutput)
        dsize = (width, height)
        img = cv2.resize(frame, dsize)
        newName = video_filename.split(".")[0] + "_static_thumbnail.jpg"
        cv2.imwrite(outputFolder + newName, img)
        break


def predictBrisque(image_path):
    img = cv2.imread(image_path)
    brisqueScore = brisque.score(img)

    return brisqueScore

def get_blur_degree(image_file, sv_num=10):
    img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    u, s, v = np.linalg.svd(img)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv/total_sv

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
        face_cascade = cv2.CascadeClassifier(haarXml)
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            if biggestFace < h:
                biggestFace = h

    elif faceDetModel == mtcnnStr:
        detector = MTCNN()
        img = cv2.imread(image)
        faces = detector.detect_faces(img)

        for result in faces:
            x, y, w, h = result['box']
            size = h
            if biggestFace < h:
                biggestFace = h

    elif faceDetModel == dnnStr:
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        img = cv2.imread(image)
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces = net.forward()
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                height = y1 - y
                if biggestFace < height:
                    biggestFace = height

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

def above_zero_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
    if x <=0:
        raise argparse.ArgumentTypeError("%r not above zero"%(x,))
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
    main()
