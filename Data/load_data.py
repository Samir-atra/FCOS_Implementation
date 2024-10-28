
def load_images(data_path):
        
    images_dict = dict()
    image_list = []
    counter = 0
    # load the images
    for file in sorted(os.listdir(train_imgs_path)):
        if counter < 600:
            image_path = os.path.join(train_imgs_path, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (800, 1024))
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            if len(image.shape) < 3:
                continue
            # plt.imshow(image)
            # plt.show()
            file_key = str(file).split('.')[0]
            imageid = re.findall("[0-9]+", file_key)
            imageid = str(imageid[1]).lstrip("_0")
            images_dict[f'{imageid}'] = image           # create a dictionary of the image titles and themselves
            image_list.append(image)
            counter += 1
        else:
            break
    return image_list




def load_labels(file_path):
# create labels lists

    labels_list = []
    bounding_boxes = []
    counter = 0
    dont_care = [0,0,0,0,0,0,0,0,0]
    annotation_path = os.path.join(annotations, "instances_train2014.json")
    boxes = dict()
    counter = 0

    with open(annotation_path, "r") as f:
        json_obj = json.load(f)
        sorted_imageids = sorted(json_obj["annotations"], key=itemgetter("image_id"))
    # draw the bounding boxes
        for line in sorted_imageids:      # iterate over the lines of the label file, where each line is a box with it's label
            if counter < 500:
                if not boxes.get(line["image_id"]):
                    boxes[line["image_id"]] = [concat(line["bbox"], line["category_id"])]
                elif boxes.get(line["image_id"]):
                    boxes[line["image_id"]] = concat(boxes[line["image_id"]], concat(line["bbox"], line["category_id"]))
                counter += 1

        print(boxes)

