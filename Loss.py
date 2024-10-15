import tensorflow as tf


def IOULoss(pred, g_label):

    pred_left = pred[0]
    pred_top = pred[1]
    pred_right = pred[2]
    pred_bottom = pred[3]

    g_label_left = g_label[0]
    g_label_top = g_label[1]
    g_label_right = g_label[2]
    g_label_bottom = g_label[3]

    target_area = (g_label_left + g_label_right) * (g_label_top + g_label_bottom)
    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

    w_intersect = tf.math.minimum(pred_left, g_label_left) + tf.math.minimum(pred_right, g_label_right)
    g_w_intersect = tf.math.maximum(pred_left, g_label_left) + tf.math.maximum(pred_right, g_label_right)
    h_intersect = tf.math.minimum(pred_bottom, g_label_bottom) + tf.math.minimum(pred_top, g_label_top)
    g_h_intersect = tf.math.maximum(pred_bottom, g_label_bottom) + tf.math.maximum(pred_top, g_label_top)
    ac_union = g_w_intersect * g_h_intersect + 1e-7
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect

    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_union - area_union) / ac_union

    
    