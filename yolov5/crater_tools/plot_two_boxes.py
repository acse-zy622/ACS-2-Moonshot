import numpy as np
import cv2
import torch
import os
import platform
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import increment_path

args = sys.argv
#/content/gdrive/MyDrive/acds_group/yolov5/runs/detect/exp
#/content/gdrive/MyDrive/acds_group/datasets/mars/labels/train2023
# print(args[1])
# print(args[2])

source_directory_img_path = str(FILE.parents[1]) + '/runs/detect/exp'
source_directory_label_path = str(FILE.parents[1]) + '/datasets/' + args[2] + '/labels/mytest' 
target_directory_path = args[1] + '/images'

# Directories
target_save_dir = increment_path(Path(target_directory_path) / 'bbox2')
target_save_dir.mkdir(parents=True, exist_ok=True)

#Coordinate conversion, originally stored in YOLOv5 format
# Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # print(x.dtype)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

import os
def draw_label(image_path,label_path):
    with open(label_path, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        # print(lb)
    # read the image file
    img = cv2.imread(str(image_path))
    # cv2.imshow('show',img)
    # print(str(image_path))
    h, w = img.shape[:2]
    lb[:, 1:] = xywhn2xyxy(lb[:, 1:], w, h, 0, 0)  #
    #print(lb)

    # line thickness
    thickness_line = max(round(sum(img.shape) / 2 * 0.003), 2)
    print("thickness is: ", thickness_line)
    # Plot
    for _, x in enumerate(lb):
        class_label = int(x[0])  # class

        cv2.rectangle(img, (int(x[1]), int(x[2])), (int(x[3]), int(x[4])), (0, 255, 0), thickness=thickness_line)
        # cv2.putText(img, str(class_label), (int(x[1]), int(x[2] - 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
        #             color=(0, 0, 255), thickness=2)
    return     img

if __name__ == '__main__':
    for root, dirs, files in os.walk(source_directory_img_path):
            for f in files:
                file_name = f.split('.')[0]+".csv"
                image_path = os.path.join(source_directory_img_path, f)
                label_path =os.path.join(source_directory_label_path, file_name)
                target =os.path.join(target_save_dir, f)
                img= draw_label(image_path, label_path)
                # print(target)
                cv2.imwrite(target, img)
