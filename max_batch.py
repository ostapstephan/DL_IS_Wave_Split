
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Flatten

import tensorflow as tf
import os
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm
import gc
from IPython import embed

AUTOTUNE = tf.data.experimental.AUTOTUNE

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


TOTAL_VAR = 0

def _parse_batch(record_batch, sample_rate, duration):
    n_samples = sample_rate * duration

    # Create a description of the features
    feature_description = {
        'audio': tf.io.FixedLenFeature([n_samples], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)
    return example['audio'], example['audio']

def get_dataset_from_tfrecords(tfrecords_dir='tfrecords', split='train', batch_size=64, sample_rate=8000, duration=5):
    if split not in ('train', 'test', 'validate'):
        raise ValueError("split must be either 'train', 'test' or 'validate'")

    # List all *.tfrecord files for the selected split
    pattern = os.path.join(tfrecords_dir, f'{split}*.tfrecord')
    files_ds = tf.data.Dataset.list_files(pattern)

    # Disregard data order in favor of reading speed
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)

    # Read TFRecord files in an interleaved order
    ds = tf.data.TFRecordDataset(files_ds,compression_type='ZLIB',num_parallel_reads=8)
    # load batch size examples
    ds = ds.batch(batch_size)

    # Parse a batch into a dataset of [audio, label] pairs
    ds = ds.map(lambda x: _parse_batch(x, sample_rate, duration))

    # Repeat the training data for n_epochs. Don't repeat test/validate splits.
    # if split == 'train':
        # ds = ds.repeat(n_epochs)
    return ds.prefetch(buffer_size=AUTOTUNE)


def main():
    # Train
    batch = 1024
    tf.config.set_soft_device_placement(True) #runs on cpu if gpu isn't availible
    train_ds = get_dataset_from_tfrecords(tfrecords_dir='/share/audiobooks/tf_records',batch_size=batch)

    for i in train_ds:
        # model = Model(tf.shape(tf.expand_dims(i[0],2)),7,2)
        break

    # model.model(tf.expand_dims(i[0],2))
    # embed()
    # split into input (X) and output (y) variables
    X = np.random.uniform(0,1,[batch,40000])
    y = np.random.uniform(0,1,[batch,40000])


    # define the keras model
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, input_dim=40000, activation='relu'))
    model.add(tf.keras.layers.Dense(400, activation='relu'))
    model.add(tf.keras.layers.Dense(40000, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    for r in train_ds:
        break
    model.build(r[0].shape)
    model.summary()

    model.fit(train_ds, epochs=1, verbose=1)
    # make class predictions with the model
    # predictions = model.predict_classes(X)
    # summarize the first 5 cases





if __name__ == '__main__':
    main()
