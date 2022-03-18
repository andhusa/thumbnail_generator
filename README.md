# thumbnail_generator
# Installation
You need Python version 3 before you start the installation.
The environment needed, is possible to install by running the command ```pip install -r requirements.txt```.
# Get started
The thumbnail_generator is possible to run in terminal by running ```python create_thumbnail.py``` and giving its belonging parameters.
Running ```python create_thumbnail.py -h``` gives detailed information on which parameters it is possible to give.

To run a GUI instead of running the ```create_thumbnail.py``` file in terminal with parameters, it is possible to write the command ```python ats_interface.py``` and a GUI window will appear.

```
create_thumbnail.py [-h] [-LEliteserien2019 | -LSoccernet | -xl] [-CSurma | -xc] [-IQAOcampo | -xi] [-BSVD | -BLaplacian | -xb] [-dlib | -haar | -mtcnn | -dnn | -xf] [-cuthr CLOSEUPTHRESHOLD]
                           [-brthr BRISQUETHRESHOLD] [-logothr LOGOTHRESHOLD] [-svdthr SVDTHRESHOLD] [-lapthr LAPLACIANTHRESHOLD] [-css CUTSTARTSECONDS] [-ces CUTENDSECONDS]
                           [-nfe NUMBEROFFRAMESTOEXTRACT | -fre FRAMERATETOEXTRACT | -fpse FPSEXTRACT] [-ds DOWNSCALEPROCESSINGIMAGES] [-dso DOWNSCALEOUTPUTIMAGE] [-as ANNOTATIONSECOND] [-bac BEFOREANNOTATIONSECONDSCUT]
                           [-aac AFTERANNOTATIONSECONDSCUT] [-st STATICTHUMBNAILSEC]
                           destination
                           ```
