"""
implementation for the FCOS model, the coompiler 
and the training loop
"""

import tensorflow as tf

from src.model.heads import head
from src.model.backbone import backbone
from src.model.pyramid import FPN


class FCOS(tf.keras.Model):
    """
    implementation for the FCOS model using the 
    ResNet50 as a backbone
    """
    def __init__(self):
        super(FCOS, self).__init__()
        self.backbone = backbone()
        self.pyramid = FPN()
        self.bias = tf.keras.initializers.Constant(-tf.math.log((1 - 0.01) / 0.01))
        self.class_num = 80
        self.classification_head = head(self.class_num, self.bias)
        self.centerness_head = head(1, self.bias)
        self.box_head = head(4, "zero")
        
    def call(self, images, training=False):
        """
        the full fcos model assembled parts 
        
        Args:
            images: input images tensor
            training: boolean, whether in training mode
            
        Returns:
            dict: dictionary of all the outputs of the model heads 
        """
        c3, c4, c5 = self.backbone(images, training=training)
        p3_out, p4_out, p5_out, p6_out, p7_out = self.pyramid(c3, c4, c5)
        
        classifier_out = []
        box_out = []
        centerness_out = []
        
        for layer in [p3_out, p4_out, p5_out, p6_out, p7_out]:
            classifier_out.append(self.classification_head(layer))
            centerness_out.append(self.centerness_head(layer))
            box_out.append(self.box_head(layer))
        
        classifier_out = tf.concat(classifier_out, axis=1, name="classifier")
        centerness_out = tf.concat(centerness_out, axis=1, name="centerness")
        box_out = tf.concat(box_out, axis=1, name="box")
        
        return {"classifier": classifier_out, "centerness": centerness_out, "box": box_out}



# loss function
# loss function
# focal = tf.keras.losses.CategoricalFocalCrossentropy(
#     alpha = 0.25,
#     gamma = 2.0,
# )
# iouloss = IOULoss()
# bce = tf.keras.losses.BinaryCrossentropy()

