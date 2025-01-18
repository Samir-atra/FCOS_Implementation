"""
implementation for the FCOS model, the coompiler 
and the training loop
"""

import tensorflow as tf
from loss import IOULoss
from heads import head


# model backbone
# FCOS uses ResNeXt not his one for better performance but this one to begine with.
def backbone():
    """
    imaplementation of the ResNet backbone model pre-trained 
    on the ImageNet dataset.
    
    Returns:
        c3: the output of the third resnet block
        c4: the output of the forth resnet block
        c5: the output of the fifth resent block
    """
    
    resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(800, 1024, 3),
            classes=80,
        )

    c3, c4, c5 = resnet.get_layer("conv3_block4_out").output, resnet.get_layer("conv4_block6_out").output, resnet.get_layer("conv5_block3_out").output
    # backbone_model = tf.keras.model(input  = resnet.input,
    #                                 output = [c3, c4, c5])
    
    return c3, c4, c5


def pyramid(c3, c4, c5):
    """
        the feature pyramid network implementation
        
    Args:
        c3 : the third resnet block output
        c4 : the fourth resnet block output
        c5 : the fifth resnet block outout 

    Returns:
        [p3_out, p4_out, p5_out, p6_out, p7_out] (list): list of the outputs of 
        the third forth fifth sixth seventh layers of the feature pyramid network
    """
    
    conv3_1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
    conv4_1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
    conv5_1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
    conv3_3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
    conv4_3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
    conv5_3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
    conv6_3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
    conv7_3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
    upsample = tf.keras.layers.UpSampling2D(2)


    p3_out = conv3_1(c3)
    p4_out = conv4_1(c4)
    p5_out = conv5_1(c5)
    p4_out = p4_out + upsample(p5_out)
    p3_out = p3_out + upsample(p4_out)
    p3_out = conv3_3(p3_out)
    p4_out = conv4_3(p4_out)
    p5_out = conv5_3(p5_out)
    p6_out = conv6_3(c5)
    p7_out = conv7_3(tf.keras.layers.relu(p6_out))

    return [p3_out, p4_out, p5_out, p6_out, p7_out]


def fcos():
    """
    the full fcos model assembled parts 
    
    Returns:
        list of all the outputs of the model heads 
        for one training example
    """
    
    classifier_out = []
    box_out = []
    centerness_out = []
    class_num = 80
    
    c3, c4, c5 = backbone()
    p3_out, p4_out, p5_out, p6_out, p7_out = pyramid(c3, c4, c5)

    bias = tf.constant_initializer(-tf.math.log((1 - 0.01) / 0.01))

    classification_head = head(class_num, bias)
    centerness_head = head(1, bias)
    box_head = head(4, "zero")

    for layer in [p3_out, p4_out, p5_out, p6_out, p7_out]:
        classifier_out.append(classification_head(layer))
        centerness_out.append(centerness_head(layer))
        box_out.append(box_head(layer))
    
    classifier_out = tf.concat(classifier_out, axis = 1)
    centerness_out = tf.concat(centerness_out, axis = 1)
    box_out = tf.concat(box_out, axis = 1)
    
    return tf.concat([classifier_out, centerness_out, box_out], axis = -1)

 







# IOU loss

iouloss = IOULoss.iou_loss()




# loss function

focal = tf.keras.losses.CategoricalFocalCrossentropy(
    alpha = 0.25,
    gamma = 2.0,
)

fcosloss = iouloss + focal



model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01,
              weight_decay = 0.0001,
              momentum = 0.9),
              loss = tf.keras.loss()

              )


