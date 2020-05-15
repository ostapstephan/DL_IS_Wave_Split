import argparse
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf

# Thanks to
# https://towardsdatascience.com/how-to-build-efficient-audio-data-pipelines-with-tensorflow-2-0-b3133474c3c1
# for making this easy to do

_BASE_DIR = os.path.dirname('/home/car-sable/libre_data/')
_loc_BASE_DIR = os.path.dirname('/home/car-sable/libre_data/')

_DEFAULT_META_CSV = os.path.join(_loc_BASE_DIR, 'meta_id.csv')
_DEFAULT_OUTPUT_DIR = os.path.join(_BASE_DIR, 'tf_records/')

_DEFAULT_DURATION = 5  # seconds
_DEFAULT_SAMPLE_RATE = 8000 

_DEFAULT_TEST_SIZE = 0.1
_DEFAULT_VAL_SIZE = 0.1

# For a 50gb dataset, this makes it so that the avg shard is about 100 mb
_DEFAULT_NUM_SHARDS_TRAIN = 400
_DEFAULT_NUM_SHARDS_TEST = 50
_DEFAULT_NUM_SHARDS_VAL = 50

# _DEFAULT_NUM_SHARDS_TRAIN = 8
# _DEFAULT_NUM_SHARDS_TEST = 1
# _DEFAULT_NUM_SHARDS_VAL = 1



_SEED = 67083 

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class TFRecordsConverter:
    """Convert audio to TFRecords."""
    def __init__(self, meta, output_dir, n_shards_train, n_shards_test,
                 n_shards_val, duration, sample_rate, test_size, val_size):
        self.output_dir = output_dir
        self.n_shards_train = n_shards_train
        self.n_shards_test = n_shards_test
        self.n_shards_val = n_shards_val
        self.duration = duration
        self.sample_rate = sample_rate

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        df = pd.read_csv(meta)

        # Shuffle data by "sampling" the entire data-frame
        self.df = df.sample(frac=1, random_state=_SEED)

        n_samples = len(df)
        # test_size and val_size is between 0 and 1 
        # n_test is an integer
        self.n_test = math.ceil(n_samples * test_size)
        self.n_val = math.ceil(n_samples * val_size)
        self.n_train = n_samples - self.n_test - self.n_val

    def _get_shard_path(self, split, shard_id, shard_size):
        # the :03d is for zero padding
        return os.path.join(self.output_dir, f'{split}-{shard_id:03d}-{shard_size}.tfrecord')

    def _write_tfrecord_file(self, shard_path, indices):
        """Write TFRecord file."""

        with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
            for index in indices:
                file_path = self.df.loc_of_wav8k.iloc[index]
                label = self.df.id.iloc[index]

                raw_audio = tf.io.read_file(file_path)
                audio, sample_rate = tf.audio.decode_wav(
                    raw_audio,
                    desired_channels=1,  # mono
                    desired_samples=-1 ) # self.sample_rate * self.duration)

                # Example is a flexible message type that contains key-value
                # pairs, where each key maps to a Feature message. Here, each
                # Example contains two features: A FloatList for the decoded
                # audio data and an Int64List containing the corresponding
                # label's index.
                example = tf.train.Example(features=tf.train.Features(feature={
                    'audio': _float_feature(audio.numpy().flatten().tolist()),
                    'label': _int64_feature(label)}))

                out.write(example.SerializeToString())

    def convert(self):
        """Convert to TFRecords.

        Partition data into training, testing and validation sets. Then,
        divide each data set into the specified number of TFRecords shards.
        """
        splits = ('train', 'test', 'validate')
        split_sizes = (self.n_train, self.n_test, self.n_val)
        split_n_shards = (self.n_shards_train, self.n_shards_test,
                          self.n_shards_val)

        offset = 0
        for split, size, n_shards in zip(splits, split_sizes, split_n_shards):
            print('Converting {} set into TFRecord shards...'.format(split))
            shard_size = math.ceil(size / n_shards)
            cumulative_size = offset + size

            for shard_id in range(1, n_shards + 1):
                step_size = min(shard_size, cumulative_size - offset)
                shard_path = self._get_shard_path(split, shard_id, step_size)
                # Generate a subset of indices to select only a subset of
                # audio-files/labels for the current shard.
                file_indices = np.arange(offset, offset + step_size)
                self._write_tfrecord_file(shard_path, file_indices)
                offset += step_size

        print('Number of training examples: {}'.format(self.n_train))
        print('Number of testing examples: {}'.format(self.n_test))
        print('Number of validation examples: {}'.format(self.n_val))
        print('TFRecord files saved to {}'.format(self.output_dir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--meta-data-csv', type=str, dest='meta_csv',
                        default=_DEFAULT_META_CSV,
                        help='File containing audio file-paths and '
                             'corresponding labels. (default: %(default)s)')

    parser.add_argument('-o', '--output-dir', type=str, dest='output_dir',
                        default=_DEFAULT_OUTPUT_DIR,
                        help='Output directory to store TFRecord files.'
                             '(default: %(default)s)')

    parser.add_argument('--num-shards-train', type=int,
                        dest='n_shards_train',
                        default=_DEFAULT_NUM_SHARDS_TRAIN,
                        help='Number of shards to divide training set '
                             'TFRecords into. (default: %(default)s)')

    parser.add_argument('--num-shards-test', type=int,
                        dest='n_shards_test',
                        default=_DEFAULT_NUM_SHARDS_TEST,
                        help='Number of shards to divide testing set '
                             'TFRecords into. (default: %(default)s)')

    parser.add_argument('--num-shards-val', type=int,
                        dest='n_shards_val',
                        default=_DEFAULT_NUM_SHARDS_VAL,
                        help='Number of shards to divide validation set '
                             'TFRecords into. (default: %(default)s)')

    parser.add_argument('--duration', type=int,
                        dest='duration',
                        default=_DEFAULT_DURATION,
                        help='The duration for the resulting fixed-length '
                             'audio-data in seconds. Longer files are '
                             'truncated. Shorter files are zero-padded. '
                             '(default: %(default)s)')

    parser.add_argument('--sample-rate', type=int,
                        dest='sample_rate',
                        default=_DEFAULT_SAMPLE_RATE,
                        help='The _actual_ sample-rate of wav-files to '
                             'convert. Re-sampling is not yet supported. '
                             '(default: %(default)s)')

    parser.add_argument('--test-size', type=float,
                        dest='test_size',
                        default=_DEFAULT_TEST_SIZE,
                        help='Fraction of examples in the testing set. '
                             '(default: %(default)s)')

    parser.add_argument('--val-size', type=float,
                        dest='val_size',
                        default=_DEFAULT_VAL_SIZE,
                        help='Fraction of examples in the validation set. '
                             '(default: %(default)s)')

    return parser.parse_args()


def main(args):
    converter = TFRecordsConverter(args.meta_csv,
                                   args.output_dir,
                                   args.n_shards_train,
                                   args.n_shards_test,
                                   args.n_shards_val,
                                   args.duration,
                                   args.sample_rate,
                                   args.test_size,
                                   args.val_size)
    converter.convert()


if __name__ == '__main__':
    main(parse_args())







