import cv2
import os
from moviepy.editor import *
#from tensorflow import keras
#from keras.preprocessing.image import ImageDataGenerator


folder_path = "../videoAndOutput/"
#model = keras.models.load_model(folder_path + 'models/thumbnail_vs_no_thumbnail.h5')
video_filename = "athc1dz9jiyfy.ts"
video_path = folder_path + video_filename
frames_folder = folder_path + video_filename.split(".")[0] + "_frames"

def main():
    print("PREDICTING")

    #clip = VideoFileClip(video_path)
    #print(clip.duration)
    #ffmpeg_extract_subclip(video_path, 0, 10, targetname= folder_path + "test.mp4")
    # Read the video from specified path
    cam = cv2.VideoCapture(video_path)
    
    try:
        # creating a folder named data

        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)
            print("created folder: " + frames_folder)
    
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
            name = frames_folder + '/frame' + str(currentframe) + '.jpg'
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
    '''
    #dir is your directory path as string
    test_data_generator = ImageDataGenerator(rescale=1./255)
    IMAGE_SIZE = 200
    TEST_SIZE = len(next(os.walk(frames_folder))[2]) 
    IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
    test_generator = test_data_generator.flow_from_directory(
        frames_folder,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=1,
        class_mode="binary", 
        shuffle=False)
    probabilities = model.predict_generator(test_generator, TEST_SIZE)

    for index, probability in enumerate(probabilities):
        image_path = frames_folder + "/" + test_generator.filenames[index]
        #img = mpimg.imread(image_path)
        #plt.imshow(img)
        if probability > 0.5:
            print(image_path)
            print("Probability: " + str(probability[0]*100) + " thumbnail")
            #plt.title("%.2f" % (probability[0]*100) + "% dog")
        else:
            print(image_path)
            print("Probability: " + str((1-probability[0])*100) + " no-thumbnail")
            #plt.title("%.2f" % ((1-probability[0])*100) + "% cat")
        #plt.show()
        '''
    pass


if __name__ == "__main__":
    main()