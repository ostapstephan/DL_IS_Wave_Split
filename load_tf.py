import os
import tensorflow as tf
import numpy as np

from IPython import embed
AUTOTUNE = tf.data.experimental.AUTOTUNE


def _parse_batch(record_batch, sample_rate, duration):
    n_samples = sample_rate * duration

    # Create a description of the features
    feature_description = {
        'audio': tf.io.FixedLenFeature([n_samples], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)

    return example['audio'], example['label']


def get_dataset_from_tfrecords(tfrecords_dir='tfrecords', split='train',
                               batch_size=64, sample_rate=8000, duration=5):

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
    train_ds = get_dataset_from_tfrecords(tfrecords_dir='/share/audiobooks/tf_records')
    '''
    lol = []
    i= 0
    for record in train_ds:
        i+=1
        temp = record[1].numpy()
        
        lol.append( np.sum(temp[:-1] - temp[1:]== 0)/len(temp))
        if i %10==0:
            print(i)
            print(sum(lol)/len(lol))
        # embed()

    # print(train_ds)
    print(sum(lol)/len(lol))
    '''
    
    # model = tf.keras.models.load_model('model.h5')
    # model.fit(train_ds, epochs=10)


if __name__ == '__main__':
    main()
