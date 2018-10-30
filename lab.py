import tensorflow as tf
import numpy as np

logits = np.ones([1,4,4,1], dtype=np.float32)
logits[:,0,:,:] = 0
logits[:,:,0,:] = 0


masks = np.ones([1,4,4,1], dtype=np.float32)
masks[:,3,:,:] = 0
masks[:,:,3,:] = 0
tf_logits = tf.constant(logits, dtype=tf.float32)
tf_masks = tf.constant(masks, dtype=tf.float32)

# prediction = tf.cast(logits > 0.7, tf.float32)

# _3_dim_max = tf.argmax(tf_pred, 2)
# _3_dim_mask = tf.argmax(tf_mask, 2)
# correct_pred = tf.equal(tf_pred, tf_mask)
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
Iou_val, update = tf.metrics.mean_iou(labels=tf_masks, predictions=tf_logits, num_classes=2)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    sess.run([update])
    print(tf_masks.eval())
    print("0________________0")
    print(tf_logits.eval())
    print("- metrics:{}".format(Iou_val.eval()))
