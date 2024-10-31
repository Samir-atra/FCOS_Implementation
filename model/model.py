# model backbone
# FCOS uses ResNeXt not his one.
tfm.vision.backbones.ResNet(
    model_id = 50,
    depth_multiplier = 1.0,
    replace_stem_max_pool = False,
    scale_stem = True,
    activation = "relu",
    bn_trainable = True)

tf.keras.applications.ResNet50(
    include_top=True,
    weights=&#x27;imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation=&#x27;softmax'
)



tfm.vision.decoders.FPN(
    input_specs: Mapping[str, tf.TensorShape],
    min_level = 3,
    max_level = 7,
    num_filters = 256,
    fusion_type = 'sum',
    use_separable_conv = False,
    use_keras_layer = False,
    activation = 'relu',
    use_sync_bn = False,
    norm_momentum = 0.99,
    norm_epsilon = 0.001,
    kernel_initializer = 'VarianceScaling',
    kernel_regularizer = None,
    bias_regularizer = None,
)



# IOU loss

def unitboxIOU(X, X_hat):
  for (i,j):
    if X_hat != 0:





# loss function

focal = tf.keras.losses.CategoricalFocalCrossentropy(
    alpha = ,
    gamma = ,
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