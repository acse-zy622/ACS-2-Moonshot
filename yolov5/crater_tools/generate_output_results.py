import shutil
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from utils.general import increment_path

args = sys.argv
# args[1] : output path user-specificed
# print("api-2: ",args[2])
print("\n")
print("******************* your output data is collecting ******************* ")
print("\n")

# targets subdirectories prepare
detections_dir = args[1] + '/detections'
image_dir = args[1] + '/images/bbox1'
image_dir_par = args[1] + '/images'
statistics_dir = args[1] + '/statistics'

# if old test file exsit
if  os.path.exists(detections_dir):
    shutil.rmtree(detections_dir)

if  os.path.exists(image_dir):
    shutil.rmtree(image_dir)

if  os.path.exists(statistics_dir):
    shutil.rmtree(statistics_dir)

# detections files (bbox list) yolov5/results ???? -> input_path/detections
source_detections_dir = args[1] + '/results' # /Users/yd1522/ACSE-2/example-data


# images files copy input_path/images -> yolov5/datasets/test/images
source_image_dir = str(FILE.parents[1]) + '/runs/detect/exp'

# statistics copy 
source_statistics_dir = str(FILE.parents[1]) + 'runs/val/test_val/exp/' + 'val-test-statistics-results.csv'# test file path

# copy
# Directories

shutil.copytree(source_detections_dir, detections_dir)
shutil.copytree(source_image_dir, image_dir)
shutil.copy(source_statistics_dir, statistics_dir)



# some information for user
print("                  detections labels is in path:                ")
print(detections_dir)


print("\n")
print("                  images with bounding boxes in path:           ")

print(image_dir_par)

print("\n")
print("                  statistics results over the whole dataset:           ")

print(statistics_dir)

print("\n")
print("*******************  output results collecting  done! *******************")
