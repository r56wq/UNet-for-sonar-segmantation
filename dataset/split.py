import os
import random
import sys

# setting up the ratio
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

os.system("rm -rf ./train ./test ./val")

# Create output directories
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join("./", split, "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join("./", split, "Annotations"), exist_ok=True)

all_jpeg_names = os.listdir(os.path.join("./", "JPEGImages"))

# shuffle images
random.shuffle(all_jpeg_names)

all_count = len(all_jpeg_names)
train_count = int(all_count * train_ratio)
test_count = int(all_count * test_ratio)
val_count = int(all_count * val_ratio)

# copy files
for img in all_jpeg_names[0:train_count]:
    Jpeg_src = os.path.join("./", "JPEGImages", img)
    Jpeg_des = os.path.join("./", "train", "JPEGImages")
    Annotation_src = os.path.join("./", "Annotations", img.split(".")[0]+".png")
    Annotation_des = os.path.join("./", "train", "Annotations")
    os.system(f"cp {Jpeg_src} {Jpeg_des}")
    os.system(f"cp {Annotation_src} {Annotation_des}")
    
    
for img in all_jpeg_names[train_count:train_count + test_count]:
    Jpeg_src = os.path.join("./", "JPEGImages", img)
    Jpeg_des = os.path.join("./", "test", "JPEGImages")
    Annotation_src = os.path.join("./", "Annotations", img.split(".")[0]+".png")
    Annotation_des = os.path.join("./", "test", "Annotations")
    os.system(f"cp {Jpeg_src} {Jpeg_des}")
    os.system(f"cp {Annotation_src} {Annotation_des}")

for img in all_jpeg_names[train_count + test_count : -1]:
    Jpeg_src = os.path.join("./", "JPEGImages", img)
    Jpeg_des = os.path.join("./", "val", "JPEGImages")
    Annotation_src = os.path.join("./", "Annotations", img.split(".")[0]+".png")
    Annotation_des = os.path.join("./", "val", "Annotations")
    os.system(f"cp {Jpeg_src} {Jpeg_des}")
    os.system(f"cp {Annotation_src} {Annotation_des}")





