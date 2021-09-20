import json
import sys
import os
import time
import numpy as np
import cv2
import onnx
import onnxruntime
from onnx import numpy_helper
 
model_dir ="/home/andrehus/egne_prosjekter/videoAndOutput/models/face_detection"
model=model_dir+"/mnet_cov2.onnx"
path="/home/andrehus/egne_prosjekter/videoAndOutput/a1qepbd80z75p_frames/frames/frame1020.jpg"
 
#Preprocess the image
img = cv2.imread(path)
img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_AREA)
img.resize((1, 3, 640, 640))
 
data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(input_name)
print(output_name)
 
result = session.run([output_name], {input_name: data})
print(np.array(result).squeeze())
#prediction=int(np.argmax(np.array(result).squeeze(), axis=0))
#print(prediction)