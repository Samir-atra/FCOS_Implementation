"""implementation for the FCOS model, the coompiler 
and the training loop"""

import tensorflow as tf
from ..loss import IOULoss
# import tensorflow_models as tfm


# model backbone
# FCOS uses ResNeXt not his one for better performance but this one to begine with.
resnet = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(800, 1024, 3),
        classes=80,
    )

c3, c4, c5 = resnet.get_layer("conv3_block4_out").output, resnet.get_layer("conv4_block6_out").output, resnet.get_layer("conv5_block3_out").output
backbone_model = tf.keras.model(input  = resnet.input,
                                output = [c3, c4, c5])

###################

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
p3_out = conv3_3(p3_out)  #
p4_out = conv4_3(p4_out)  #
p5_out = conv5_3(p5_out)  #
p6_out = conv6_3(c5)   #
p7_out = conv7_3(tf.keras.layers.relu(p6_out))  #


#####################################


# IOU loss

iouloss = IOULoss.iou_loss()




# loss function

focal = tf.keras.losses.CategoricalFocalCrossentropy(
    alpha = 0.25,
    gamma = 2.0,
)

fcosloss = unitboxIOU + focal



model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01,
              weight_decay = 0.0001,
              momentum = 0.9),
              loss = tf.keras.loss()

              )






def schedule(epoch, lr):
  if epoch == 60000 or epoch == 80000:
    lr = lr / 10
    return lr

sched = tf.keras.callbacks.LearningRateSchedualer(
    schedule
)






model.fit(epochs = 90000,
          batch_size = 16,
          callbacks = [sched],

          )

