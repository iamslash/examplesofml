import tensorflow as tf
tf.set_random_seed(111)

def main():
    l_X = [1., 2., 3.]
    l_Y = [1., 2., 3.]
    t_W = tf.Variable(tf.random_normal([1]), name='W')
    t_b = tf.Variable(tf.random_normal([1]), name='b')
    t_X = tf.placeholder(tf.float32, shape=[None], name='X')
    t_Y = tf.placeholder(tf.float32, shape=[None], name='Y')
    t_H = t_X * t_W + t_b
    t_C = tf.reduce_mean(tf.square(t_H - t_Y))
    t_O = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    t_T = t_O.minimize(t_C)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n_step in range(10000):
            _, f_cost, l_W, l_b = sess.run([t_T, t_C, t_W, t_b], feed_dict={t_X: l_X, t_Y: l_Y})
            if n_step % 20 == 0:
                print(f"{n_step:10d} {f_cost:10.7f}", l_W, l_b)
        print(sess.run(t_H, feed_dict={t_X: [5]}))
        print(sess.run(t_H, feed_dict={t_X: [2.5]}))
        print(sess.run(t_H, feed_dict={t_X: [1.5, 3.5]}))

if __name__ == "__main__":
    main()