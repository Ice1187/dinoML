# import cv2
import time
import keyboard
import numpy as np
import matplotlib.pyplot as plt
from GetData import GetImg, GetKey_dino, ProcessImg_dino
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, SimpleRNN, Activation, Flatten, MaxPooling2D, TimeDistributed


def prepare():
	for i in range(5):
		print(5-i)
		time.sleep(1)
	print('Start recording!')


def process_batch(input_list, label_list, batch_size=5):
	drop = len(input_list) % batch_size
	input_list = input_list[drop:,:,:]
	label_list = label_list[drop:,:]
	input_list = input_list.reshape(len(input_list), 165, 955, 1)  # Conv2D needs ndim=4 (n_img, width, height, channels)
	return input_list, label_list


def get_img_and_key():
	img = GetImg(window_place)
	key = GetKey_dino()
	img = ProcessImg_dino(img).reshape(1, 165, 955)
	return img, key


def collect_data(seconds, mode='return', path='', epochs=1):

	if mode=='return' or mode=='r':
		last_time = time.time()
		img, key = get_img_and_key()
		img_data = img
		key_data = [key]

		while(True):
			if time.time()-last_time >= seconds:
				break
			img, key = get_img_and_key()
			img_data = np.vstack((img_data, img))
			key_data.append(key)
			# cv2.imshow('Gray dino', img)
			# if cv2.waitKey(25) & 0xFF == ord('q'):
			# 	break
			# count_time += used_time
		key_data = np.array(key_data)

		return img_data, key_data

	elif mode=='collect' or mode=='c':
		print('[-] Start collecting training data.')

		for i in range(epochs):
			last_time = time.time()
			img, key = get_img_and_key()
			img_data = img
			key_data = [key]
			while(True):
				img, key = get_img_and_key()
				img_data = np.vstack((img_data, img))
				key_data.append(key)

				if time.time() - last_time >= seconds:
					key_data = np.array(key_data)
					np.savez(path, img_data=img_data, key_data=key_data)
					print("[+] {} seconds has passed.\n Saving Training data...".format((i+1)*seconds))
					break

		print('[-] Collection process has finished.')


def send_data(key):
	key = np.argmax(key)
	if key == 0:
		print('Nothing')
	elif key == 1:
		# keyboard.press('space')
		# # time.sleep(1)
		# keyboard.release('space')
		keyboard.send('space')
		print('key: space')
	elif key == 2:
		# keyboard.press('down')
		# # time.sleep(1)
		# keyboard.release('down')
		keyboard.send('down')
		print('key: down')



def build_cnn(input_list, label_list, epochs=10):
	model = Sequential()

	model.add(Conv2D(
		batch_input_shape=(size_batch, 165, 955, 1),
		filters=32,
		kernel_size=(3,3),
		padding='same',
		activation='relu'))

	model.add(MaxPooling2D(
		pool_size=(2,2)))

	model.add(Conv2D(
		filters=16,
		kernel_size=(3,3),
		padding='same',
		activation='relu'))

	model.add(MaxPooling2D(
		pool_size=(2,2)))

	model.add(Conv2D(
		filters=8,
		kernel_size=(3,3),
		padding='same',
		activation='relu'))

	model.add(MaxPooling2D(
		pool_size=(2,2)))

	model.add(Flatten())

	model.add(Dense(
		3,
		activation='softmax'))

	print(model.summary())

	model.compile(
		loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['acc'])

	model.fit(input_list, label_list, epochs=epochs, batch_size=size_batch, verbose=2)

	return model


def save_model(model):
	while(True):
		save_flag = input('Save this model? (y/n): ')

		if save_flag == 'Y' or save_flag == 'y':
			model_name = input('Model name: ')
			if not ('.h5' in model_name):
				model_name = model_name + '.h5'
			model.save(model_name)
			break

		elif save_flag == 'n' or save_flag == 'N':
			print('Run without saving the model.')
			break

		else:
			continue


# Set parameters
window_place = (0, 250, 955, 415)
img_shape = (165, 955, 1)
size_batch = 5
seconds = 30.0
model_path = ''
data_path = './traing_data_2.npz'

# data_1 -> lack of little tree, can get about 150 (10 sec)
# data_2 -> keep get loss=0.3, which sometimes occur on data_1, might be the sign of unbalanced data

# prepare()
# collect_data(seconds, mode='c', path=data_path, epochs=1)
# input_list, label_list = collect_data(seconds, mode='r', path=data_path, epochs=10)
data = np.load(data_path)
input_list, label_list = np.reshape(data['img_data'], (-1,165,955,1)), data['key_data']
input_list, label_list = process_batch(input_list, label_list, batch_size=size_batch)
print(input_list.shape)
print(label_list.shape)

# build model
model = build_cnn(input_list, label_list, epochs=10)
save_model(model)
# model = load_model(model_path)


# Play
prepare()
while(True):
	img = ProcessImg_dino(GetImg(window_place)).reshape(1, 165, 955, 1)
	move = model.predict(img)
	send_data(move)



















# model = Sequential()
# model.add(TimeDistributed(Conv2D(
# 	filters = 30,
# 	kernel_size = (3,3),
# 	input_shape=(img_shape),
# 	padding='same',
# 	activation='relu')))

# # print(model.output_shape)

# model.add(TimeDistributed(MaxPooling2D(
# 	pool_size=(2,2),
# 	padding='valid')))

# # print(model.output_shape)

# model.add(TimeDistributed(Conv2D(
# 	filters = 10,
# 	kernel_size = (3,3),
# 	padding='same',
# 	activation='relu')))

# # print(model.output_shape)

# model.add(TimeDistributed(MaxPooling2D(
# 	pool_size=(2,2),
# 	padding='valid')))


# # print(model.output_shape)

# model.add(SimpleRNN(
# 	units=10))

# model.add(Dense(3))
# model.add(Activation('softmax'))

# # print(model.summary())

# model.compile(
# 	optimizer='Adadelta',
# 	loss='sparse_categorical_crossentropy',
# 	metrics=['acc']
# 	)

# model.fit(input_list, label_list, epochs=10)
