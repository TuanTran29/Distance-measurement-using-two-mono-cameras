# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import rospy
import math 
from std_msgs.msg import String

# pub = rospy.Publisher('/cameraCommand', String, queue_size=10)
# rospy.init_node('talker', anonymous=True)

# def talker(locX,locY):

#     # while not rospy.is_shutdown():
#     if (locX < 250):
#         hello_str = "97"
#     elif (locX > 350):
#         hello_str = "100"
#     elif ((locX >= 250) & (locX <= 350)):
#         hello_str = "119"
#     else :
#         hello_str = "1"
#     rospy.loginfo(hello_str)
#     print(hello_str)
#     pub.publish(hello_str)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-y", "--yolo", required=True,
	help="base path to hello_strYOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# capture frames from a camera
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

while(1):
	# reads frames from a camera
	ret, image = cap.read()
	ret, image2 = cap2.read()
	(H, W) = image.shape[:2]
	(H2, W2) = image2.shape[:2]
	# print("Dimension {:f} x {:f}".format(H,W))
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	# # show timing information on YOLO
	print("[INFO] YOLO1 took {:.6f} seconds".format(end - start))
	# # initialize our lists of detected bounding boxes, confidences, and
	# # class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				# print("Center {:.0f} x {:.0f}".format(centerX,centerY))
				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	blob2 = cv2.dnn.blobFromImage(image2, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob2)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	# show timing information on YOLO
	print("[INFO] YOLO2 took {:.6f} seconds".format(end - start))
	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes2 = []
	confidences2 = []
	classIDs2 = []
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W2, H2, W2, H2])
				(centerX, centerY, width, height) = box.astype("int")
				# print("Center {:.0f} x {:.0f}".format(centerX,centerY))
				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes2.append([x, y, int(width), int(height)])
				confidences2.append(float(confidence))
				classIDs2.append(classID)
	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs2 = cv2.dnn.NMSBoxes(boxes2, confidences2, args["confidence"],
		args["threshold"])

	P1 = 0
	P2 = 0
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			# print(classIDs[i])
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# print("Location: ",x,", ",y)
			# print("Width: " ,w ,", Height ",h)
			if (classIDs[i] == 39):
				CenterX1 = x+w/2
				P1 = 1280 - CenterX1                
				print("Center1 {:.0f}".format(CenterX1))
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
				cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2) 

			# 	talker(x,y)
			# draw a bounding box rectangle and label on the image

# ensure at least one detection exists
	if len(idxs2) > 0:
		# loop over the indexes we are keeping
		for i in idxs2.flatten():
			# extract the bounding box coordinates
			# print(classIDs2[i])
			(x, y) = (boxes2[i][0], boxes2[i][1])
			(w, h) = (boxes2[i][2], boxes2[i][3])
			# print("Location: ",x,", ",y)
			# print("Width: " ,w ,", Height ",h)
			if (classIDs2[i] == 39):
				CenterX2 = x+w/2
				P2 = CenterX2
				print("Center2 {:.0f}".format(CenterX2))
				color = [int(c) for c in COLORS[classIDs2[i]]]
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs2[i]], confidences2[i])
				cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2)        

	if ((len(idxs2) > 0) & (len(idxs) > 0)) :
		A = 25
		H1 = 1280  
		H2 = 640   
		omega1 = 75
		omega2 = 25  
		beta1 = 52.5   
		beta2 = 77.5  
		sin1 =  P1 * omega1 / H1 + beta1
		sin2 =  P2 * omega2 / H2 + beta2
		sin3 =  180 - (sin1 + sin2)             
		h = A * math.sin(math.radians(sin1)) * math.sin(math.radians(sin2)) / math.sin(math.radians(sin3))
		print("Distance {:.4f}cm\n".format(h))                                        

	# show the output image
	cv2.imshow("Image", image)
	# cv2.imshow("Image", image2)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break



# if __name__ == '__main__':
#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass
  
# Close the window
cap.release()
  
# De-allocate any associated memory usage
cv2.destroyAllWindows() 
