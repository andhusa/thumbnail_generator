import cv2
import os
import re
from moviepy.editor import *
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import imutils
from os.path import isfile, join


folder_path = "../videoAndOutput/"
model = keras.models.load_model(folder_path + 'models/thumbnail_vs_no_thumbnail.h5')
video_filename = "athc1dz9jiyfy.ts"
video_path = folder_path + video_filename
frames_folder = folder_path + video_filename.split(".")[0] + "_frames"

def main():
    print("PREDICTING")

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
            print ('Creating: ' + name)
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
        currentframe += 1
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    predictAndRemove()
    fileName = selectMean()
    print(fileName)



def predictAndRemove():
    
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
    
        if probability > 0.77:
            print(image_path)
            print("Probability: " + str(probability[0]*100) + " thumbnail")
            img = cv2.imread(image_path)
            img = imutils.resize(img, width=500)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(gray.shape)
            (mean, blurry) = detect_blur_fft(gray)
            print("mean: " + str(mean))
            print("blurry: " + str(blurry))
            if blurry:
                os.remove(image_path)
        elif probability > 0.5:
            #print(image_path)
            #print("Not clear thumbnail")
            #print("Probability: " + str(probability[0]*100) + " thumbnail")
            os.remove(image_path)
        else:
            #print(image_path)
            #print("Probability: " + str((1-probability[0])*100) + " no-thumbnail")
            os.remove(image_path)

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



def detect_blur_fft(image, size=60, thresh=5, vis=False):
	# grab the dimensions of the image and use the dimensions to
	# derive the center (x, y)-coordinates
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))
    # compute the FFT to find the frequency transform, then shift
	# the zero frequency component (i.e., DC component located at
	# the top-left corner) to the center where it will be more
	# easy to analyze
	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)
    # check to see if we are visualizing our output
	if vis:
		# compute the magnitude spectrum of the transform
		magnitude = 20 * np.log(np.abs(fftShift))
		# display the original input image
		(fig, ax) = plt.subplots(1, 2, )
		ax[0].imshow(image, cmap="gray")
		ax[0].set_title("Input")
		ax[0].set_xticks([])
		ax[0].set_yticks([])
		# display the magnitude image
		ax[1].imshow(magnitude, cmap="gray")
		ax[1].set_title("Magnitude Spectrum")
		ax[1].set_xticks([])
		ax[1].set_yticks([])
        # zero-out the center of the FFT shift (i.e., remove low
	# frequencies), apply the inverse shift such that the DC
	# component once again becomes the top-left, and then apply
	# the inverse FFT
	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)
    # compute the magnitude spectrum of the reconstructed image,
	# then compute the mean of the magnitude values
	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)
	# the image will be considered "blurry" if the mean value of the
	# magnitudes is less than the threshold value
	return (mean, mean <= thresh)

if __name__ == "__main__":
    main()
