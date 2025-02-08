"""
the back bone used for the feature pyramid network implementation
which is the RsNet50 model pre-trained on the ImageNet dataset
"""


import tensorflow as tf

# model backbone
# FCOS uses ResNeXt not his one for better performance but this one to begin with.

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
    
    return c3, c4, c5
