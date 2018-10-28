import tensorflow as tf
import numpy as np

logits = np.ones([2,3,3,1], dtype=np.float32) * 0.5
logits[1,2,1,0] = 1
logits[1,1,2,0] = 1

mask = np.ones([2,3,3,1], dtype=np.float32)

tf_logits = tf.constant(logits, dtype=tf.float32)
tf_mask = tf.constant(mask, dtype=tf.float32)

probs = tf.nn.sigmoid(tf_logits)
prediction = tf.cast(probs > 0.7, tf.float32)

# _3_dim_max = tf.argmax(tf_pred, 2)
# _3_dim_mask = tf.argmax(tf_mask, 2)
# correct_pred = tf.equal(tf_pred, tf_mask)
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(probs.eval())
    print(prediction.eval())
