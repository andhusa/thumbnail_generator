import cv2
import os
from moviepy.editor import *
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

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
        # creating a folder named data

        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)
            os.makedirs(frames_folder + "/frames")
            print("created folder: " + frames_folder + "/frames")
    
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    
    # frame
    currentframe = 0
    frame_skip = 60
    while(True):
        # reading from frame
        ret,frame = cam.read()
        if not ret:
            break
        if currentframe % frame_skip == 0:
            # if video is still left continue creating images
            name = frames_folder + '/frames/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
        currentframe += 1
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    predict()



def predict():
    
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
        image_path = frames_folder + "/" + test_generator.filenames[index]
        i += 1
        print("")
        print("")
        print("PHOTO NUMBER " + str(i))
    
        if probability > 0.5:
            print(image_path)
            print("Probability: " + str(probability[0]*100) + " thumbnail")
            (mean, blurry) = detect_blur_fft(image_path)
            print("mean: " + str(mean))
            print("blurry: " + str(blurry))
        else:
            print(image_path)
            print("Probability: " + str((1-probability[0])*100) + " no-thumbnail")
            os.remove(image_path)

def detect_blur_fft(image, size=60, thresh=10, vis=False):
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
