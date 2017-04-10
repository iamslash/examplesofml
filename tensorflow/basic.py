
def test():
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)

    hello = tf.constant("Hello World")
    sess = tf.Session()
    print(sess.run(hello))


   
if __name__ == "__main__":
    test()
