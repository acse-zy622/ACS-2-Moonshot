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
# print("running generate_test_file.py", args[1])

dir1 = str(FILE.parents[1]) + '/datasets/mars/images/mytest'
dir2 = str(FILE.parents[1]) + '/datasets/mars/labels/mytest'

# if old test file exsit
if  os.path.exists(dir1):
    shutil.rmtree(dir1)

if  os.path.exists(dir2):
    shutil.rmtree(dir2)

# images files copy input_path/images -> yolov5/datasets/test/images
source_dir = args[1] + '/images' # /Users/yd1522/ACSE-2/example-data
destination_dir = str(FILE.parents[1]) + '/datasets/mars/images/mytest'  # /Users/yd1522/ACSE-2/group_project/yolov5/datasets/test/images

# print(destination_dir)

source_dir_label = args[1] + '/labels' # test file path
destination_dir_label = str(FILE.parents[1]) + '/datasets/mars/labels/mytest' 

# copy
# Directories
# target_save_dir = increment_path(Path(destination_dir)/ 'detect')
# target_save_dir.mkdir(parents=True, exist_ok=True)

shutil.copytree(source_dir, destination_dir)
shutil.copytree(source_dir_label, destination_dir_label)


# test yaml file
# copy .yaml file

source_dir_yaml = str(FILE.parents[1]) + '/data/mars.yaml'
destination_dir_yaml = str(FILE.parents[1]) + '/data/mars-detect.yaml'

if  os.path.exists(destination_dir_yaml):
    os.remove(destination_dir_yaml)
else:
   shutil.copy(source_dir_yaml, destination_dir_yaml)

## change new .yaml file
line_number = 14  # Whatever the line number you're trying to replace is
original_line ="val: images/val2023  # val images (relative to 'path') 128 images"
replacement_line = "val: images/mytest  # val images (relative to 'path') 128 images"

items = os.listdir(".")  # Gets all the files & directories in the folder containing the script
file_name = destination_dir_yaml

with open(file_name, 'r') as f:
  d = f.read().replace(original_line, replacement_line)
with open(file_name, 'w') as f:
  f.write(d)

for i in range(3):
    print("\n")
print("******************* test data loading done!*******************")