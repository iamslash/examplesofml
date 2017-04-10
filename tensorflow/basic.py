
def test():
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)

    hello = tf.constant("Hello World")
    sess = tf.Session()
    #print(sess.run(hello))

    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)
    node3 = tf.add(node1, node2)
    print("node1:", node1, "node2:", node2)
    print("node3:", node3)

    sess = tf.Session()
    print("sess.run(node1, node2): ", sess.run([node1, node2]))
    print("sess.run(node3): ", sess.run(node3))

    # placeholder
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
    print(sess.run(adder_node, feed_dict = {a: [1, 3], b: [2, 4]}))

    # Variables
    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
    biases  = tf.Variable(tf.zeros([200]), name="biases")
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
   
if __name__ == "__main__":
    test()
