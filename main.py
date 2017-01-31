import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SEED = 0xbad1dea
EPOCHS = 20000
BATCH_SIZE = 50



def random_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, seed=SEED))


def constant_biases(shape, value):
    return tf.Variable(tf.constant(value=value, shape=shape))


def conv2d(input, weights):
    return tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


print("Loading MNIST dataset...")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

in_vector = tf.placeholder("float", [None, 28 * 28])  # flattened 28x28 matrix

c1_weights = random_weights([5, 5, 1, 8])
c1_biases = constant_biases([8], 0.1)

c2_weights = random_weights([7, 7, 8, 16])
c2_biases = constant_biases([16], 0.1)

dense_weights = random_weights([7*7*16, 1024])
dense_biases = constant_biases([1024], 0.1)

output_weights = random_weights([1024, 10])
output_biases = constant_biases([10], 0.1)

conv1 = max_pool(tf.nn.relu(conv2d(tf.reshape(in_vector, [-1, 28, 28, 1]), c1_weights) + c1_biases))
conv2 = max_pool(tf.nn.relu(conv2d(conv1, c2_weights) + c2_biases))
dense_output = tf.nn.relu_layer(tf.reshape(conv2, [-1, 7*7*16]), dense_weights, dense_biases)
out_vector = tf.nn.relu_layer(dense_output, output_weights, output_biases)

expected = tf.placeholder("float", [None, 10])

cross_enthropy = tf.nn.softmax_cross_entropy_with_logits(logits=out_vector, labels=expected)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_enthropy)

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

print("Training network...")
correct = tf.equal(tf.argmax(out_vector, 1), tf.argmax(expected, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
for i in range(EPOCHS):
    if i%50 == 0:
        print(f"epoch {i+1}/{EPOCHS}")
    batch_input, batch_output = mnist.train.next_batch(BATCH_SIZE)
    session.run(train_step, {in_vector: batch_input, expected: batch_output})
    if i%1000 == 0:
        accuracy_value = accuracy.eval(feed_dict={in_vector: batch_input, expected: batch_output})
        print(f"Accuracy: {accuracy_value}")

accuracy_value = accuracy.eval(feed_dict={in_vector: mnist.test.images, expected: mnist.test.labels})
print(f"Accuracy: {accuracy_value}")
