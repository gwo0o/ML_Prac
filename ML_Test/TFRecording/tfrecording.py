import numpy as np
import tensorflow as tf

n_examples = 10
n_features = 10

examples = np.random.randint(0, 100, size = (n_examples, n_features))

for idx in range(examples.shape[0]):
    print("label={}, feature={}".format(idx, examples[idx]))

with tf.io.TFRecordWriter('example.tfrecord') as tfrecord:
    for idx in range(examples.shape[0]):
        label = [idx]
        feature = examples[idx]
        features = {
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = label)),
            'feature': tf.train.Feature(int64_list = tf.train.Int64List(value = feature))
        }
        example = tf.train.Example(features = tf.train.Features(feature = features))
        tfrecord.write(example.SerializeToString())

def map_fn(serialized_example):
    feature = {
        'label': tf.io.FixedLenFeature([1], tf.int64),
        'feature': tf.io.FixedLenFeature([n_features], tf.int64)
    }
    example = tf.io.parse_single_example(serialized_example, feature)
    return example['feature'], example['label']

dataset = tf.data.TFRecordDataset('example.tfrecord')
dataset = dataset.map(map_fn)

for feature, label in dataset.take(10):
    print(feature)

print(data1.as_numpy_iterator())
print(list(data1.as_numpy_iterator()))
print(list(data1.as_numpy_iterator())[0])
print(list(data1.as_numpy_iterator())[0][0])
print(list(list(data1.as_numpy_iterator())[0][0]))
print([list(list(data1.as_numpy_iterator())[0][0])])