import tensorflow as tf

def main():
    tf.set_random_seed(111)
    X = [1, 2, 3]
    Y = [1, 2, 3]
    W = tf.Variable(tf.random_normal([1]), name='W')
    b = tf.Variable(tf.random_normal([1]), name='b')
    H = X * W + b
    C = tf.reduce_mean(tf.square(H - Y))
    O = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    T = O.minimize(C)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(2000):
            sess.run(T)
            if step % 20 == 0:
                f_cost = sess.run(C)
                f_W = sess.run(W)
                f_b = sess.run(b)

                #print("step = {:7d} loss = {:5.3f} W = {:5.3f} b = {:5.3f}".format(step, f_cost, f_W, f_b) )
                print(step, f_cost, f_W, f_b)

if __name__ == "__main__":
    main()
