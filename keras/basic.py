def main(_epochs):
    import keras
    import numpy

    x = numpy.array([0, 1, 2, 3, 4])
    y = 2 * x + 1

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_shape=(1,)))
    model.compile('SGD', 'mse')

    model.fit(x[:3], y[:3], epochs=2000, verbose=0)

    print("Tgts: ", y[3:])
    print("Pred: ", model.predict(x[3:]).flatten())

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('USAGE) python basic.py [epochs]')
        sys.exit()
    main(sys.argv[1])