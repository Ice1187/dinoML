import cv2
import keyboard
import numpy as np
from PIL import ImageGrab


def GetImg(window_place):
	img = np.array(ImageGrab.grab(window_place))	# left, top, right, bottom
	return img


def ProcessImg_dino(img): 
	trash, processed_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
	processed_img = cv2.Canny(processed_img, threshold1=50, threshold2=100)

	# [!] Consider whthere houghline/drawline is needed  [!]
	# lines = cv2.HoughLinesP(processed_img, 1.0, np.pi/180, 100, 10, 15)
	# print(lines[0])
	# draw_lines(processed_img, lines)
	# print(processed_img.ndim, key)
	
	return processed_img


def GetKey_dino():
	key = [1, 0, 0]  # nothing, space, down
	if keyboard.is_pressed('space'):
		key = [0, 1, 0]
	elif keyboard.is_pressed('down'):
		key = [0, 0, 1]
	
	return key


# def draw_lines(img, lines):
# 	# pass
# 	try:
# 		for line in lines:
# 			coords = line[0]
# 			cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 55,255], 1)
# 	except:
# 		pass		



if __name__ == '__main__':
	pass