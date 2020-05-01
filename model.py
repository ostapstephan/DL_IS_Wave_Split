import os
import tensorflow as tf
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm

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

#########################################################################
######################## MODEL LAYERS GO BELOW ##########################
#########################################################################

def normalize_batch(batch):
    '''
    this is used to normalize the TF record so that we have two
    equal volume audio tracks
    [batch_size, sample_len_in_sec*sample_rate]
    '''
    mean = tf.repeat(tf.expand_dims(tf.math.reduce_mean(batch,axis = 1),axis = 1),tf.shape(batch)[1],axis =-1 )
    std =  tf.repeat(tf.expand_dims(tf.math.reduce_std(batch,axis = 1), axis = 1),tf.shape(batch)[1],axis =-1 )
    out = (batch - mean)/(2*std)
    print(tf.shape(out))
    return out

def dense( inputs , weights,leaky_relu_alpha = 0.2, dropout_rate = 0.5):
    x = tf.nn.leaky_relu( tf.matmul( inputs , weights ) , alpha=leaky_relu_alpha )
    return tf.nn.dropout( x , rate=dropout_rate )

def sep_conv(inputs):
    out = tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter,
    strides, padding, data_format=None, dilations=None, name=None)
    return out

def conv2d( inputs , filters , stride_size ):
    out = tf.nn.conv2d( inputs , filters , strides=[ 1 , stride_size , stride_size , 1 ] ,
    padding=padding )
    return tf.nn.leaky_relu( out , alpha=leaky_relu_alpha )

def conv1d( inputs , filter , stride_size,padding='SAME',dilations=None):
    return tf.nn.conv1d(input, filters, stride=stride_size, padding=padding, dilations=dilations)


def parametric_relu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32,
                        trainable=True)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg

def conv1dBlock(inputs,filters,strid_size):
    #     conv1 = conv1d(inputs ...)
    #     prelu1 = parametric_relu(conv1)
    #     norm1 = tf.keras.layers.LayerNormalization(...)(prelu1)
    #     dconv1 = conv1d(norm1 ....)
    #     prelu2 = parametric_relu(dconv1)
    #     norm2 = tf.keras.layers.LayerNormalization(...)(prelu2)
    #     output = conv1d(norm2 ...)
    #     skip = conv1d(norm2 ...)
    #
    #     return output,skip
    '''
    high level overview:

    1x1 conv
    prelu
    normalize
    d_conv (nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,groups=hidden_channel,padding=self.padding)) : CODE FOR DCONV FROM PAPER IMPLEMENTATION
    prelu
    normalize
    1x1 conv into skip: 1x1 into output

    On first pass generate weights list
    '''
    return

tf.keras.layers.LayerNormalization(
    axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
    gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    beta_constraint=None, gamma_constraint=None, trainable=True, name=None,)


# conv1 = conv1d(layer0)
#
# return out


def mix_audio(audio, epoch):
    # record = shape[data,label]
    # data = shape[batch_size,40000]
    return tf.add(audio,tf.roll(audio,1+epoch,axis=0))

#########################################################################
#########################  MODEL LAYERS ABOVE  ##########################
#########################################################################


# '''
class Model(object):
    def __init__(self):
        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
        self.epoch = 0
        self.num_epochs = 1

        self.shapes = [
            # [ 3 , 3 , 3 , 16 ] ,  # conv shape
            [ 40000 , 4000 ],
            # [ 4000 , 4000 ],
            # [ 4000 , 4000 ],
            # [ 4000 , 4000 ],
            # [ 4000 , 4000 ],
            # [ 4000 , 4000 ],
            # [ 4000 , 8000 ],
            # [ 4000 , 8000 ],
            [ 4000 , 2000 ],
            [ 2000, 80000 ]
        ]

        learning_rate = 0.01
        self.optimizer = tf.optimizers.Adam( learning_rate )

        def get_weight( shape , name ):
            return tf.Variable( self.initializer( shape ) , name=name , trainable=True , dtype=tf.float32 )

        self.initializer = tf.initializers.glorot_uniform()
        self.weights = []

        for i in range( len( self.shapes ) ):
            '''
            model definition:
                loop on stack:
                    generate blocks -> return shapes
                create the weights based on shapes
            '''
            self.weights.append( get_weight( self.shapes[ i ] , 'weight{}'.format( i ) ) )

        print('\n\n\n\n')
        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
        print(tf.compat.v1.trainable_variables())

        print('\n\n\n\n')



    def loss(self, pred , target ):
        label = tf.reshape(tf.concat([target, tf.roll(target,1+self.epoch,axis=0)],axis = 1), [-1,2,40000] )
        return tf.losses.mean_squared_error(label, pred )

    def model(self,x):
        mixed = mix_audio(x,self.epoch)
        d = dense( mixed , self.weights[0] )
        for i in range(1,len(self.shapes)):
            d = dense( d , self.weights[i] )

        # print(tf.shape(d))
        streams  = tf.reshape( d , shape=( -1  ,2, 40000 ))

        return streams

    def train_step( self, model, inputs ):
        with tf.GradientTape() as tape:
            current_loss = self.loss( self.model( inputs ), inputs )
        grads = tape.gradient( current_loss , self.weights )
        self.optimizer.apply_gradients( zip( grads , self.weights ) )
        # print( tf.reduce_mean( current_loss ).numpy() )

    def fit(self, dataset):
        for self.epoch in range( self.num_epochs ):

            # i = 0
            for features in tqdm(dataset):
                # i+=1
                audio, label = features[0] , features[1]
                self.train_step( self.model , audio  )

                # if i == 50 :
                    # break


def main():
    # Train
    tf.config.set_soft_device_placement(True)

    train_ds = get_dataset_from_tfrecords(tfrecords_dir='/share/audiobooks/tf_records')
    model = Model()

    model.fit(train_ds)


    # Test/ Create Audios

    for record in train_ds:
        out = model.model(record[0]).numpy()
        for i in range(out.shape[0]):
            # print(out[i])
            scipy.io.wavfile.write(f'mixed/b{i}.wav', 8000, out[i][0])
            scipy.io.wavfile.write(f'mixed/a{i}.wav', 8000, record[0][i].numpy())
        break

    '''
    for epoch in range(5):
        for record in :
            norm = normalize_batch(record[0]).numpy()
            # audio_mix = mix_audio(norm,epoch).numpy()
            # for i in range(audio_mix.shape[0]):
                # scipy.io.wavfile.write(f'mixed/a{i}.wav', 8000, audio_mix[i])
            # embed() # dont forget to break out of a loop if using this
            break
        break

    # model = tf.keras.models.load_model('model.h5')
    # model.fit(train_ds, epochs=10)
    # '''


if __name__ == '__main__':
    main()
