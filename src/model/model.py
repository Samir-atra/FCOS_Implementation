"""
implementation for the FCOS model, the coompiler 
and the training loop
"""

import tensorflow as tf
from loss import IOULoss
from loss import FcosLoss
from heads import head
from backbone import backbone
from pyramid import FPN


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
        
    def call(self):
        """
        the full fcos model assembled parts 
        
        Returns:
            list of all the outputs of the model heads 
            for one training example
        """
        c3, c4, c5 = self.backbone()
        p3_out, p4_out, p5_out, p6_out, p7_out = self.pyramid(c3, c4, c5)
        
        classifier_out = []
        box_out = []
        centerness_out = []
        
        for layer in [p3_out, p4_out, p5_out, p6_out, p7_out]:
            classifier_out.append(self.classification_head(layer))
            centerness_out.append(self.centerness_head(layer))
            box_out.append(self.box_head(layer))
        
        classifier_out = tf.concat(classifier_out, axis = 1, name = "classifier")
        centerness_out = tf.concat(centerness_out, axis = 1, name = "centerness")
        box_out = tf.concat(box_out, axis = 1, name = "box")
        
        return tf.concat([classifier_out, centerness_out, box_out], axis = -1)



# loss function
focal = tf.keras.losses.CategoricalFocalCrossentropy(
    alpha = 0.25,
    gamma = 2.0,
)
iouloss = IOULoss()
bce = tf.keras.losses.BinaryCrossentropy()

model = FCOS()

model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01,
              weight_decay = 0.0001,
              momentum = 0.9),
              loss = {'classifier': focal, 'box': iouloss, 'centerness': bce},
              metrics = ['precision'])


def schedule(epoch, lr):
  if epoch == 60000 or epoch == 80000:
    lr = lr / 10
    return lr

sched = tf.keras.callbacks.LearningRateScheduler(
    schedule
)

model.fit(
          epochs = 90000,
          batch_size = 16,
          callbacks = [sched],
          )

