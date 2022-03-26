# thumbnail_generator
# Installation
You need Python version 3 before you start the installation.
The environment needed, is possible to install by running the command ```pip install -r requirements.txt```.
# Get started
The thumbnail generator is possible to run in terminal by running ```python create_thumbnail.py``` and giving its belonging parameters.
Running ```python create_thumbnail.py -h``` gives detailed information on which parameters it is possible to give.

To run a GUI instead of running the ```create_thumbnail.py``` file in terminal with parameters, it is possible to write the command ```python ats_interface.py``` and a GUI window will appear.

It is possible to receive a thumbnail when running the command ```python create_thumbnail.py <path_to_video_file>``` in terminal. This will run with the default parameters, but it is also possible to modify the parameters with the flags that are available.

The flags that are available for the ```create_thumbnail.py``` file is shown when running the command ```python create_thumbnail.py -h``` and the following will display in terminal:

```
create_thumbnail.py [-h] [-LEliteserien2019 | -LSoccernet | -xl] [-CSurma | -xc] [-IQAOcampo | -xi] [-BSVD | -BLaplacian | -xb] [-dlib | -haar | -mtcnn | -dnn | -xf] [-cuthr CLOSEUPTHRESHOLD]
                           [-brthr BRISQUETHRESHOLD] [-logothr LOGOTHRESHOLD] [-svdthr SVDTHRESHOLD] [-lapthr LAPLACIANTHRESHOLD] [-css CUTSTARTSECONDS] [-ces CUTENDSECONDS]
                           [-nfe NUMBEROFFRAMESTOEXTRACT | -fre FRAMERATETOEXTRACT | -fpse FPSEXTRACT] [-ds DOWNSCALEPROCESSINGIMAGES] [-dso DOWNSCALEOUTPUTIMAGE] [-as ANNOTATIONSECOND] [-bac BEFOREANNOTATIONSECONDSCUT]
                           [-aac AFTERANNOTATIONSECONDSCUT] [-st STATICTHUMBNAILSEC]
                           destination
                           
Thumbnail generator

positional arguments:
  destination           Destination of the input to be processed. Can be file or folder.

options:
  -h, --help            show this help message and exit
  -LEliteserien2019     Surma model used for logo detection, trained on Eliteserien 2019.
  -LSoccernet           Surma model used for logo detection, trained on Soccernet.
  -xl, --xLogoDetection
                        Don't run logo detection.
  -CSurma               Surma model used for close-up detection.
  -xc, --xCloseupDetection
                        Don't run close-up detection.
  -IQAOcampo            Ocampo model used for image quality assessment.
  -xi, --xIQA           Don't run image quality prediction.
  -BSVD                 SVD method used for blur detection.
  -BLaplacian           Laplacian method used for blur detection.
  -xb, --xBlurDetection
                        Don't run blur detection.
  -dlib                 Dlib detection model is slow, but presice.
  -haar                 Haar detection model is fast, but unprecise.
  -mtcnn                MTCNN detection model is slow, but precise.
  -dnn                  DNN detection model is fast and precise.
  -xf, --xFaceDetection
                        Don't run the face detection.
  -cuthr CLOSEUPTHRESHOLD, --closeUpThreshold CLOSEUPTHRESHOLD
                        The threshold value for the close-up detection model. The value must be between 0 and 1. The default is: 0.75
  -brthr BRISQUETHRESHOLD, --brisqueThreshold BRISQUETHRESHOLD
                        The threshold value for the image quality predictor model. The default is: 35
  -logothr LOGOTHRESHOLD, --logoThreshold LOGOTHRESHOLD
                        The threshold value for the logo detection model. The value must be between 0 and 1. The default value is: 0.1
  -svdthr SVDTHRESHOLD, --svdThreshold SVDTHRESHOLD
                        The threshold value for the SVD blur detection. The default value is: 0.65
  -lapthr LAPLACIANTHRESHOLD, --laplacianThreshold LAPLACIANTHRESHOLD
                        The threshold value for the Laplacian blur detection. The default value is: 1000
  -css CUTSTARTSECONDS, --cutStartSeconds CUTSTARTSECONDS
                        The number of seconds to cut from start of the video. These seconds of video will not be processed in the thumbnail selection. The default value is: 0
  -ces CUTENDSECONDS, --cutEndSeconds CUTENDSECONDS
                        The number of seconds to cut from the end of the video. These seconds of video will not be processed in the thumbnail selection. The default value is: 0
  -nfe NUMBEROFFRAMESTOEXTRACT, --numberOfFramesToExtract NUMBEROFFRAMESTOEXTRACT
                        Number of frames to be extracted from the video for the thumbnail selection process. The default is: 50
  -fre FRAMERATETOEXTRACT, --framerateToExtract FRAMERATETOEXTRACT
                        The framerate wanted to be extracted from the video for the thumbnail selection process.
  -fpse FPSEXTRACT, --fpsExtract FPSEXTRACT
                        Number of frames per second to extract from the video for the thumbnail selection process.
  -ds DOWNSCALEPROCESSINGIMAGES, --downscaleProcessingImages DOWNSCALEPROCESSINGIMAGES
                        The value deciding how much the images to be processed should be downscaled. The default value is: 0.5
  -dso DOWNSCALEOUTPUTIMAGE, --downscaleOutputImage DOWNSCALEOUTPUTIMAGE
                        The value deciding how much the output thumbnail image should be downscaled. The default value is: 1.0
  -as ANNOTATIONSECOND, --annotationSecond ANNOTATIONSECOND
                        The second the event is annotated to in the video.
  -bac BEFOREANNOTATIONSECONDSCUT, --beforeAnnotationSecondsCut BEFOREANNOTATIONSECONDSCUT
                        Seconds before the annotation to cut the frame extraction.
  -aac AFTERANNOTATIONSECONDSCUT, --afterAnnotationSecondsCut AFTERANNOTATIONSECONDSCUT
                        Seconds after the annotation to cut the frame extraction.
  -st STATICTHUMBNAILSEC, --staticThumbnailSec STATICTHUMBNAILSEC
                        To generate a static thumbnail from the video, this flag is used. The second the frame should be clipped from should follow as an argument. Running this flag ignores all the other flags.
```

