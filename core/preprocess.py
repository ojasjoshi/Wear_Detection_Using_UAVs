#!usr/bin/python
import os
import numpy as np
import cv2
from PIL import ImageOps
import random
import pdb
from copy import deepcopy

NUM_DIR = 2
DATA_DIR = '../data/'

def get_data(num_dir=NUM_DIR):
	# gives CV2 format
	originals = []
	masks = []
	dirnames = np.arange(num_dir)
	for dirname in dirnames:
		dirname = DATA_DIR+str(dirname)
		if os.path.isdir(DATA_DIR+str(dirname)):
			for i, filename in enumerate(os.listdir(str(dirname))):
				if(filename!='.DS_Store'):
					img = cv2.imread(str(dirname)+'/'+str(filename),0)
					name = filename.split('.')[0]
					if(name[-5:]=='imcor'):
						originals.append(img)
					elif(name[-3:]=='bin'):
						masks.append(img)
					else:
						pass
	return originals, masks

#deprecated
def pad(img,crop_size=500):
	# takes and returns PIL format
	width, height = img.size
	desired_size = [width//crop_size,height//crop_size]

	delta_w = desired_size[0] - new_size[0]
	delta_h = desired_size[1] - new_size[1]
	padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
	
	padded_img = ImageOps.expand(img, padding)

	return padded_img

def resize_img(img,crop_size=500):
	# takes, returns CV2 format
	width, height = img.shape[0],img.shape[1]
	desired_size = (crop_size*(width//crop_size),crop_size*(height//crop_size))

	return cv2.resize(img,(desired_size))

def resized_only_wear(originals, masks, crop_size=500):
	# gives CV2 format
	preprocessed = []
	for (ori_t,mask_t) in zip(originals,masks):
		wear_img = cv2.bitwise_and(ori_t,ori_t,mask = mask_t)
		# preprocessed.append(pad(Image.fromarray(wear_img)),crop_size)	#convert to PIL before passing pad
		preprocessed.append(resize_img(wear_img,crop_size))

	return preprocessed

def resize_original(originals,crop_size=500):
	# gives CV2 format
	preprocessed = []
	for img in originals:
		preprocessed.append(resize_img(img,crop_size))

	return preprocessed

def cutout(raw_image, contour, crop_size=500, threshold = 0.2):
	#takes cv2 format
	preprocessed_cutouts = []	# each element is [img,corresponding_contour,wear_value]
	wear_cut = []
	no_wear = []
	wear_count = 0				#total num of cutouts with high wear(wear value>0.9)
	for (img,cnt) in zip(raw_image,contour):
		cuts = []
		wid= img.shape[0]
		hei = img.shape[1]
		num_x = wid//crop_size
		num_y = hei//crop_size
		
		for x in range(num_x):
			for y in range(num_y):		
				cuts.append([x*crop_size,y*crop_size,(x+1)*crop_size,(y+1)*crop_size])

		for cut_dims in cuts:
			x1,y1,x2,y2 = cut_dims
			normed_cnt_cutout = cnt[x1:x2,y1:y2]/255
			wear_value = np.sum(normed_cnt_cutout)/(normed_cnt_cutout.shape[0]*normed_cnt_cutout.shape[1])
			if(wear_value>threshold):
				wear_count += 1
			current_cutout = img[x1:x2,y1:y2]
			preprocessed_cutouts.append([current_cutout,normed_cnt_cutout*255,wear_value])
			if(threshold<wear_value):
				# print("Wear with wear_value = ", wear_value)
				wear_cut.append([current_cutout,normed_cnt_cutout*255,wear_value])
			else:
				# print("No wear with wear_value = ", wear_value)
				no_wear.append([current_cutout,normed_cnt_cutout*255,wear_value])
	print(wear_count,"number of high wear cutouts were identified")
	return wear_cut, no_wear, preprocessed_cutouts

def process(crop_size=128,threshold=0.5):
	print("Preprocessing data...")
	originals, masks = get_data()
	resized_originals = resize_original(originals)
	# resized_wear = resized_only_wear(originals, masks, crop_size)		#cv2 format
	wear_cut, no_wear_cut, preprocessed_cutouts = cutout(originals,masks,crop_size=crop_size,threshold=threshold)
	print("{} preprocessed cutouts formed".format(len(preprocessed_cutouts)))

	return wear_cut, no_wear_cut, preprocessed_cutouts

""" TODO: Implement bias_ratio """
def Imdb(wear_cut, no_wear_cut, bias_ratio=0.5):
	""" bias_ratio is bias of wear to no_wear i.e. bias_ratio*batch_size=#wear images"""
	""" TODO: Implement bias ratio """

	""" DEPRECATED USE """
	# while(batch_size>len(wear_cut) or batch_size>len(no_wear_cut)):
	# 	print("Threshold too low for given batch size. Using a higher threshold")
	# 	threshold += 0.1
	# 	wear_cut, no_wear_cut, _ = process(threshold)	

	data_list = [wear_cut, no_wear_cut]
	sizes = [len(wear_cut), len(no_wear_cut)]
	
	batch_size = deepcopy(np.amax(sizes))
	index_to_update = deepcopy(np.argmin(sizes))

	list_to_update = deepcopy(data_list[index_to_update]) # this list doesnt change
	updated_list = deepcopy(list_to_update)

	for i in range((batch_size-len(list_to_update))// len(list_to_update)):
		updated_list += list_to_update
	# add remaining data here

	trailing_size = batch_size-len(updated_list)
	trailing_idx = random.sample(range(0,len(list_to_update)),trailing_size)
	updated_list += [list_to_update[idx] for idx in trailing_idx]

	data_list[np.argmin(sizes)] = updated_list

	[wear_cut,no_wear_cut] = data_list

	assert len(wear_cut)==batch_size
	assert len(no_wear_cut)==batch_size

	batch_images = [x1[0] for x1 in wear_cut] + [x2[0] for x2 in no_wear_cut]
	batch_labels = [x1[2] for x1 in wear_cut] + [x2[2] for x2 in no_wear_cut]

	"""WARNING: only for grey images """
	for i in range(len(batch_images)):
		batch_images[i] = np.expand_dims(batch_images[i],axis=0)

	""" DEPRECATED USE """
	# wear_idx = random.sample(range(0,len(wear_cut)),int(batch_size*bias_ratio))
	# no_wear_idx = random.sample(range(0,len(no_wear_cut)),int(batch_size*(1-bias_ratio)))

	# batch_wear = [wear_cut[idx][0] for idx in wear_idx]
	# batch_no_wear = [no_wear_cut[idx][0] for idx in no_wear_idx]
	# batch_images = batch_wear + batch_no_wear

	# batch_wear_labels = [wear_cut[idx][1] for idx in wear_idx]
	# batch_no_wear_labels = [no_wear_cut[idx][1] for idx in no_wear_idx]
	# batch_labels = batch_wear_labels + batch_no_wear_labels

	assert len(batch_images)==2*batch_size
	assert len(batch_labels)==2*batch_size

	print("Data loaded!")
	return batch_images, batch_labels

""" TODO: Include RGB in get_data"""
def main():
	threshold = 0.5
	wear_cut, no_wear_cut, _ = process(threshold)
	Imdb(wear_cut, no_wear_cut)


if __name__ == '__main__':
	main()







