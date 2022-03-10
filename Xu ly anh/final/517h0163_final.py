from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())
"""
point = {0: 'V', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4' , 6: '5' ,7: '6', 8: '7', 9: '8', 10: '9', 11: '10', 12: '0', 13: '1', 14: '2', 15: '3', 16: '4', 17: '5', 18: '6', 19: '7', 20: '8', 21: '9'}
image = cv2.imread(r"E:\testcase2.jpg")



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3,5), 0)

thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


edged = cv2.Canny(thresh, 150, 200)


cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None
# ensure that at least one contour was found
if len(cnts) > 0:
	# sort the contours according to their size in
	# descending order
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	# loop over the sorted contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		# if our approximated contour has four points,
		# then we can assume we have found the paper
		if len(approx) == 4:
			docCnt = approx
			break




questionCnts = []
#cv2.drawContours(image, cnts, -1, (0, 0, 255), 5)
for c in cnts:
	# compute the bounding box of the contour, then use the
	# bounding box to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	'''
	if w >= 40 and h>=40 and ar >= 0.8 and ar <= 1.2:

		print('ar',ar)
		print('x',x)
		print('y',y)
		print('w',w)
		print('h',h)
	'''
	# in order to label the contour as a question, region
	# should be sufficiently wide, sufficiently tall, and
	# have an aspect ratio approximately equal to 1
	if w >= 45 and h >= 45	and ar >= 0.9 and ar <= 1.2:
		questionCnts.append(c)

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers

questionCnts = contours.sort_contours(questionCnts,method="top-to-bottom")[0]
#cv2.drawContours(image, questionCnts, -1, (0, 0, 255), 5)
color = [(0, 0, 225),(0, 255, 0),(255, 0, 0),(255, 0, 255),(0, 255, 255),(255, 255, 0),(255,255,255)]
k = 0
X = 1630
Y = 70
#print(len(questionCnts))
for (q, i) in enumerate(np.arange(0, len(questionCnts), 22)):
	# sort the contours for the current question from
	# left to right, then initialize the index of the
	# bubbled answer

	cnts = contours.sort_contours(questionCnts[i:i +22])[0]
	#cv2.drawContours(image, cnts, -1, color[1], 5)

	bubbled = None
	truoc = None
	sau = None

	for (j, c) in enumerate(cnts):
		# construct a mask that reveals only the current
		# "bubble" for the question

		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		# apply the mask to the thresholded image, then
		# count the number of non-zero pixels in the
		# bubble area
		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
		total = cv2.countNonZero(mask)
		# if the current total has a larger number of total
		# non-zero pixels, then we are examining the currently
		# bubbled-in answer
		if total > 1300:
			bubbled = (total, j)
			#print('bub',bubbled)
			#print('j',bubbled[1])
			if bubbled[1] <= 11:
				truoc = (total,j)
			elif bubbled[1] > 11:
				sau = (total,j)
			#cv2.drawContours(image,cnts[bubbled[1]] , -1, (0, 0, 255), 3)

	#print('truoc',truoc,'sau',sau)
	#print(truoc[1],sau[1])
	#print(point[truoc[1]],'.',point[sau[1]])
	cv2.drawContours(image, cnts[truoc[1]], -1, (0, 0, 255), 7)
	cv2.drawContours(image, cnts[sau[1]], -1, (0, 0, 255), 7)
	#print('cnts',cnts[sau[1]])
	#print(type(point[sau[1]]))
	score = point[truoc[1]]+'.'+point[sau[1]]
	#print(score)
	cv2.putText(image,score,(1630,Y+k),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 0, 0),2)
	k+=80
	#cv2.drawContours(image, cnts[bubbled[1]], -1, (0, 255, 0), 5)
#cv2.drawContours(image,cnts , -1, (0, 0, 255), 3)
#vẽ ra
#cv2.drawContours(image,questionCnts , -1, (0, 255, 255), 3)

#cv2.namedWindow("edged", cv2.WINDOW_FREERATIO)
#cv2.imshow("edged", edged)


#cv2.namedWindow("thresh", cv2.WINDOW_FREERATIO)
#cv2.imshow("thresh", thresh)

cv2.namedWindow("image", cv2.WINDOW_FREERATIO)
cv2.imshow("image", image)



cv2.waitKey(0)
cv2.destroyAllWindows()