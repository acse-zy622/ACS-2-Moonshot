import os
    
path_prefix = os.getcwd()
sub_dir_train = 'datasets/mars/labels/train'
sub_dir_val = 'datasets/mars/labels/val'

path_train = os.path.join(path_prefix, sub_dir_train)
path_val = os.path.join(path_prefix, sub_dir_val)

with open('trainval.txt', 'w') as f:
    train_list = os.listdir(path_train)
    train_list.sort()
    for i in train_list:
        name = i.split('.')[0] + '\n'
        f.write(name)
        
with open('test.txt', 'w') as f:
    val_list = os.listdir(path_val)
    val_list.sort()
    for i in val_list:
        name = i.split('.')[0] + '\n'
        f.write(name)
