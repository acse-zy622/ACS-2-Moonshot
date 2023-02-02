import os
import random
import numpy as np
import shutil
import PIL.Image as Image

pwd = os.getcwd()
path = os.path.join(os.getcwd(), "datasets/mars/labels/train2023")
path_image = os.path.join(os.getcwd(), "datasets/mars/images/train2023")
file_list = os.listdir(path)
name_list = [i.split('_')[0] for i in file_list]
name_set = set(name_list)
new_file_list = []

file_list_val = []
file_list_train = []

for name in name_set:
    count = 0
    for i in name_list:
        if i == name:
            count += 1
    len_val = int(count * 0.3)
    file_list_part = []
    for k in file_list:
        if k.split('_')[0] == name:
            file_list_part.append(k)
    random_list = [j for j in range(count)]
    random.shuffle(random_list)
    file_arr_part = np.array(file_list_part)
    random_arr = np.array(random_list)
    file_arr_part_val = file_arr_part[random_arr[:len_val]]
    file_arr_part_train = file_arr_part[random_arr[len_val:]]
    for j in file_arr_part_val:
        file_list_val.append(os.path.join(path,j))
    for j in file_arr_part_train:
        file_list_train.append(os.path.join(path,j))

print(len(file_list), len(file_list_val) + len(file_list_train))

path_train_new = os.path.join(os.getcwd(), "datasets/mars/labels/train")
path_val_new = os.path.join(os.getcwd(), "datasets/mars/labels/val")

path_train_image_new = os.path.join(os.getcwd(), "datasets/mars/images/train")
path_val_image_new = os.path.join(os.getcwd(), "datasets/mars/images/val")

for i in file_list_val:
    new_path = os.path.join(path_val_new, i.split('/')[-1])
    shutil.copyfile(i, new_path)
    new_path_image = os.path.join(path_val_image_new, i.split('/')[-1].split('.')[0] + '.png')
    img_path = os.path.join(path_image, i.split('/')[-1].split('.')[0] + '.png')
    img = Image.open(img_path)
    img.save(new_path_image)
    
for i in file_list_train:
    new_path = os.path.join(path_train_new, i.split('/')[-1])
    shutil.copyfile(i, new_path)
    new_path_image = os.path.join(path_train_image_new, i.split('/')[-1].split('.')[0] + '.png')
    img_path = os.path.join(path_image, i.split('/')[-1].split('.')[0] + '.png')
    img = Image.open(img_path)
    img.save(new_path_image)
