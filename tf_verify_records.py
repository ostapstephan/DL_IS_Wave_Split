import struct
import tensorflow as tf

# from tensorflow import app
from absl import flags
from absl import app
import glob

# from tensorflow import flags
# from tensorflow import gfile
# from tensorflow import logging

# flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "input_data_pattern", "", "File glob defining for the TFRecords files."
)


def main(unused_argv):
    # logging.set_verbosity(tf.logging.INFO)
    path_ = FLAGS.input_data_pattern
    paths = glob.glob(path_)
    # print("Found %s files.", len(paths))
    for path in paths:
        with open(path, "rb") as f:
            first_read = True
            while True:
                length_raw = f.read(8)

                if not length_raw and first_read:
                    print("File %s has no data.", path)
                    break
                elif not length_raw:
                    print("File %s looks good.", path)
                    break
                else:
                    first_read = False

                if len(length_raw) != 8:
                    print("File ends when reading record length: " + path)
                    break
            print(length_raw)
            (length,) = struct.unpack("L", length_raw)
            # +8 to include the crc values.
            record = f.read(length + 8)
            if len(record) != length + 8:
                print("File ends in the middle of a record: " + path)
                break


if __name__ == "__main__":
    app.run(main)
