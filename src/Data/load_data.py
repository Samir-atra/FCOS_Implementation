"""load the COCO dataset by creating two lists 
images list and labels list
"""

import sys
import re
from operator import itemgetter
import json
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/samer/Desktop/Beedoo/FCOS/FCOS_Implementation/utils")
from concatenate import concat
import tensorflow as tf


train_imgs_path = "/home/samer/Desktop/Beedoo/FCOS/FCOS_Implementation/COCO2014/train2014/"
val_imgs_path = "/home/samer/Desktop/Beedoo/FCOS/FCOS_Implementation/COCO2014/val2014/"
test_imgs_path = "/home/samer/Desktop/Beedoo/FCOS/FCOS_Implementation/COCO2014/test2014/"
annotations = "/home/samer/Desktop/Beedoo/FCOS/FCOS_Implementation/COCO2014/annotations/"


class DataLoader:
    """loading the COCO dataset images and bounding boxes as labels,
    contains two methods on for each functionality"""
    def __init__(self, train_images_path, annotations_path):
        self.train_images_path = train_images_path
        self.annotations_path = annotations_path

    def load_images(self, data_path):
        """loads the images of the dataset into a list"""
        images_dict = {}
        image_list = []
        counter = 0
        # load the images
        for file in sorted(tf.io.gfile.listdir(train_imgs_path)):
            if counter < 100:
                image_path = tf.io.gfile.join(train_imgs_path, file)
                image = tf.io.read_file(image_path)
                image = tf.io.decode_image(image, dtype=tf.dtypes.float32)
                image = tf.image.resize(
                    image, (800, 1024), method="bilinear", preserve_aspect_ratio=False
                )
                if len(image.shape) < 3:
                    continue
                # plt.imshow(image)
                # plt.show()
                file_key = str(file).split(".")[0]
                imageid = re.findall("[0-9]+", file_key)
                imageid = str(imageid[1]).lstrip("_0")
                images_dict[f"{imageid}"] = (
                    image  # create a dictionary of the image titles and themselves
                )
                plt.imshow(image)
                plt.show()
                # print(image)
                image_list.append(image)
                counter += 1
            else:
                break
        return image_list

    def load_labels(self, file_path):
        """creates labels lists for ech image consisting of the image id and
        category id and the bounding box for each object in the image"""

        labels_list = []
        bounding_boxes = []
        counter = 0
        dont_care = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        annotation_path = tf.io.gfile.join(annotations, "instances_train2014.json")
        boxes = {}
        counter = 0

        with open(annotation_path, "r", encoding="utf-8") as f:
            json_obj = json.load(f)
            sorted_imageids = sorted(
                json_obj["annotations"], key=itemgetter("image_id")
            )
            # draw the bounding boxes
            for (
                line
            ) in (
                sorted_imageids
            ):  # iterate over the lines of the label file, where each line is a box with it's label
                # print(line)
                if counter < 500:                  # due to limited computation had to limit the number of boxes to be loaded
                    if not boxes.get(line["image_id"]):
                        boxes[line["image_id"]] = [
                            concat(line["bbox"], line["category_id"])
                        ]
                    elif boxes.get(line["image_id"]):
                        boxes[line["image_id"]] = concat(
                            boxes[line["image_id"]],
                            concat(line["bbox"], line["category_id"]),
                        )
                    counter += 1

        return boxes


if __name__ == "__main__":
    loader = DataLoader(train_imgs_path, annotations)
    images_list = loader.load_images(train_imgs_path)
    boxes_dictionary = loader.load_labels(annotations)
