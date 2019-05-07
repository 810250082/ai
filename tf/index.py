import tensorflow as tf

# a = tf.Variable(2.0)
# b = tf.Variable(tf.random_normal([2, 3]))
# c = a + b
# writer = tf.summary.FileWriter('./result', tf.get_default_graph())
# writer.close()


a = tf.Variable(2.0)
b = tf.Variable(tf.random_normal([2, 3]))
c = a + b
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))
    saver.save(sess, './model/model.ckpt')

# 先构建模型, 后恢复
a = tf.Variable(0.0)
b = tf.Variable(tf.random_normal([2, 3]))
c = a + b
saver = tf.train.Saver()
# 直接恢复,不用构建模型
saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
with tf.Session() as sess:
    saver.restore(sess, './model/model.ckpt')
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))