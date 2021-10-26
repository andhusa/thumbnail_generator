# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2

protoxt = "/home/andrehus/egne_prosjekter/thumbnail_generator/detectFace/deploy.prototxt.txt"
model = "/home/andrehus/egne_prosjekter/thumbnail_generator/detectFace/res10_300x300_ssd_iter_140000.caffemodel"
confidenceThr = 0.7
# construct the argument parse and parse the arguments


def detecting_face(inputImage):
	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(protoxt, model)

	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	image = cv2.imread(inputImage)
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > confidenceThr:
			print("Face detected")
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			length = endX - startX
			height = endY - startY
			print("l" + str(length) + " h " + str(height))
			if length > 50 or height > 50:
				return True
			# draw the bounding box of the face along with the associated
			# probability
			#text = "{:.2f}%".format(confidence * 100)
			#y = startY - 10 if startY - 10 > 10 else startY + 10
			#cv2.rectangle(image, (startX, startY), (endX, endY),
			#	(0, 0, 255), 2)
			#cv2.putText(image, text, (startX, y),
			#	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	return False
	# show the output image
	#filename = inputImage.split("/")[-1]

	#cv2.imwrite("output/" + filename, image)
	#cv2.imshow("Output", image)
	#cv2.waitKey(0)

if __name__ == "__main__":
	detecting_face("iron_chic.jpg")