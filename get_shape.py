
import os
import tensorflow as tf
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm

from IPython import embed
AUTOTUNE = tf.data.experimental.AUTOTUNE

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

def parametric_relu(_x,alphas):
    '''
    x_ : the tensor you wish to apply parametric_relu to
    alphas : shape of x[1:].shape
    '''
    # if x is outta a conv ie [batch x 16 x 4 channels]
    # we need shape[-2]*shape[-1]
    pos = tf.nn.relu(_x)
    neg = tf.reshape(alphas * (_x - abs(_x)) * 0.5,tf.shape(pos))
    return pos + neg

''' define entire model -> outputs a list of lists of shapes for the entire model '''
def conv1dBlock(inputs,filters,strid_size):
    #     conv1 = conv1d(inputs ...)
    #     prelu1 = parametric_relu(conv1)
    #     norm1 = tf.keras.layers.LayerNormalization(...)(prelu1)
    #     dconv1 = conv1d(norm1 ....)
    #     prelu2 = parametric_relu(dconv1)
    #     norm2 = tf.keras.layers.LayerNormalization(...)(prelu2)
    #     output = conv1d(norm2 ...)
    #     skip = conv1d(norm2 ...)
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


    for keras layers we must call
    layer.build then call
    layer.trainable_variables
    to get the trainable Variables for the
    sgd based optimizers

    for tf.nn layers we need to supply a tensor contatining the trainable
    variables and for the layer since it's a function
    '''
    return

class Conv_1D_Block(object):
    def __init__(self,input_shape,dilation=1):
        self.hyper_params= {
        'N' : 128,
        'L' : 1,
        'B' : 128,
        'H' : 256,
        'Sc': 128,
        'P' : 3,
        'X' : 7,
        'R' : 2
        }

        self.weights_shape = []
        self.input_shape=input_shape
        self.dilation = dilation
        self.initializer = tf.initializers.glorot_uniform()
        self.getShapesofBlock()
        self.weights = []

        # self.get_weights()
        self.get_weights()
        # ^use these shape

    def getShapesofBlock(self):
            # the input shape is [batch, [self.input_shape], self.hyper_params['B'] ]
            conv1 = tf.keras.layers.Conv1D(filters=self.hyper_params['H'],kernel_size = self.hyper_params['L'],padding='same')
            conv1.build(self.input_shape) # should be (batch,B,L)
            self.weights_shape.append([tf.shape(x).numpy() for x in conv1.trainable_weights] )

            prelu1 = tf.keras.layers.PReLU(shared_axes=[1])
            prelu1_inp = conv1.compute_output_shape(self.input_shape)
            prelu1.build(prelu1_inp)
            self.weights_shape.append([tf.shape(prelu1.trainable_weights).numpy()])

        with tf.device('/CPU:0'):
            LayerNorm1 = tf.keras.layers.LayerNormalization(axis=0,dtype='float32')
            LayerNorm1_inp = prelu1.compute_output_shape(prelu1_inp)
            LayerNorm1.build(LayerNorm1_inp)
            self.weights_shape.append([LayerNorm1]) # append the whole layer

        with tf.device('/CPU:0'):
            dconv1 = tf.keras.layers.Conv1D(filters=self.hyper_params['H'], kernel_size = self.hyper_params['L'], padding = 'same') #This should be a depthwise conv but they didnt do it so neither will we
            dconv1_inp = LayerNorm1.compute_output_shape(LayerNorm1_inp)
            dconv1.build(dconv1_inp)
            self.weights_shape.append([tf.shape(x).numpy() for x in dconv1.trainable_weights])

            prelu2 = tf.keras.layers.PReLU(shared_axes=[1])
            prelu2_inp = dconv1.compute_output_shape(dconv1_inp)
            prelu2.build(prelu2_inp)
            self.weights_shape.append([tf.shape(prelu2.trainable_weights).numpy()])

        with tf.device('/GPU:0'):
            LayerNorm2 = tf.keras.layers.LayerNormalization(axis=0,dtype='float32')
            LayerNorm2_inp = prelu2.compute_output_shape(prelu2_inp)
            LayerNorm2.build(LayerNorm2_inp)
            self.weights_shape.append([LayerNorm2]) # appending the whole layer

        with tf.device('/CPU:0'):
            # skip layer
            conv2_1 = tf.keras.layers.Conv1D(filters=self.hyper_params['Sc'], kernel_size = self.hyper_params['L'], padding='same') #to skip connection
            conv2_1_inp = LayerNorm2.compute_output_shape(LayerNorm2_inp)
            conv2_1.build(conv2_1_inp)
            self.output_shape = conv2_1.compute_output_shape(conv2_1_inp)
            self.weights_shape.append([tf.shape(x).numpy() for x in conv2_1.trainable_weights])

            # output
            conv2_2 = tf.keras.layers.Conv1D(filters=self.hyper_params['B'],  kernel_size = self.hyper_params['L'], padding='same')
            conv2_2_inp = LayerNorm2.compute_output_shape(LayerNorm2_inp)
            conv2_2.build(conv2_2_inp)
            self.skip_shape = conv2_2.compute_output_shape(conv2_2_inp)
            self.weights_shape.append([tf.shape(x).numpy() for x in conv2_2.trainable_weights])

        return

    def get_weights(self):
        global TOTAL_VAR
        for w_s_l in self.weights_shape: # weight_shape_list
            if not isinstance(w_s_l[0],tf.keras.layers.Layer):
                a = [tf.Variable(self.initializer(w),trainable=True,dtype=tf.float32) for w in w_s_l ]
                self.weights.append( a   )
                TOTAL_VAR += sum([tf.reduce_prod(tf.shape(x)).numpy() for x in a ])
                print( [(tf.shape(x)).numpy() for x in a] )
            else:
                a = [w_s_l[0].trainable_weights]
                self.weights.append(a)
                TOTAL_VAR += sum([tf.reduce_prod(tf.shape(x)).numpy() for x in a ])
                print( [(tf.shape(x)).numpy() for x in a] )

    def forward_pass_block(self,input_tensor):

        print(tf.shape(self.weights[0][0]),'input shape_for conv 1d')
        conv1x1_1 = tf.nn.conv1d(input_tensor,self.weights[0][0],stride=1,padding="SAME",dilations=self.dilation)
        conv1x1_1_b = tf.add(conv1x1_1,self.weights[0][1]) #TODO add the rest of the biases to the model explicitly

        prelu1 = parametric_relu(conv1x1_1_b, self.weights[1])
        LayerNorm1 = self.weights_shape[2][0](prelu1)
        # weights[2] == keras_layers.get_trakinable_weights
        deconv1 = tf.nn.conv1d(LayerNorm1, self.weights[3][0], stride=1, padding="SAME")
        #Should be depthwise seperable convolution not transpose
        deconv1_b = tf.add(deconv1,self.weights[3][1])

        prelu2 = parametric_relu(deconv1, self.weights[4])
        LayerNorm2 = self.weights_shape[5][0](prelu2) # weights[5] == keras_layers.get_trainable_weights
        conv1x1_2_skip = tf.nn.conv1d(LayerNorm2,self.weights[6][0],stride=1,padding="SAME") #to skip connection
        conv1x1_2_skip_b = tf.add(conv1x1_2_skip,self.weights[6][1])

        conv1x1_3_output = tf.nn.conv1d(LayerNorm2,self.weights[7][0],stride=1,padding="SAME")
        conv1x1_3_output_b = tf.add(conv1x1_3_output,self.weights[7][1])

        return conv1x1_2_skip_b, conv1x1_3_output_b

class Coder(object):
    def __init__(self,input_shape):
        self.Block = Conv_1D_Block(object)

    def getShapesofBlock(self):
        return self.Block.getShapesofBlock()

    def forward_pass(self,input_tensor):
        skip, output = self.Block.forward_pass_block()
        return output

def mix_audio(audio, epoch):
    # record = shape[data,label]
    # data = shape[batch_size,40000]
    return tf.add(audio,tf.roll(audio,1+epoch,axis=0))

#########################################################################
#########################  MODEL LAYERS ABOVE  ##########################
#########################################################################


class SeparationStack(object):
    def __init__(self,input_shape, Height,Width,hyper_params = None):

        if isinstance(hyper_params ,type(None) ):
            self.hyper_params= {
            'N' : 128,
            'L' : 1,
            'B' : 128,
            'H' : 256,
            'Sc': 128,
            'P' : 3,
            'X' : 7,
            'R' : 2
            }

        self.Blocks = []
        self.weights = [] #will be the weights of the layers specific to the overall separation stack not of the individual blocks
        self.weights_shape = []
        self.initializer = tf.initializers.glorot_uniform()
        self.height = Height
        self.width = Width
        self.input_shape = input_shape
        shape = self.getShapesofbeginningBlock()


        for i in range(Width):
            add = []
            for j in range(Height):
                block_to_add = Conv_1D_Block(shape,2**j)
                add.append(block_to_add)
                shape = block_to_add.output_shape
            self.Blocks.append(add)

        self.getShapesofendBlock()

        self.get_weights()


    def get_weights(self):
        global TOTAL_VAR
        for w_s_l in self.weights_shape: # weight_shape_list
            if not isinstance(w_s_l[0],tf.keras.layers.Layer):
                a =[tf.Variable(self.initializer(w),trainable=True,dtype=tf.float32) for w in w_s_l ]
                self.weights.append( a )
                TOTAL_VAR += sum([tf.reduce_prod(tf.shape(x)).numpy() for x in a ])
            else:
                a = [w_s_l[0].trainable_weights]
                self.weights.append( a )
                TOTAL_VAR += sum([tf.reduce_prod(tf.shape(x)).numpy() for x in a ])

        temp= []
        for i in range(self.width):
            for j in range(self.height):
                temp.append(self.Blocks[i][j].weights)

        self.weights = self.weights[:-2] + temp + self.weights[-2:]


    def getShapesofbeginningBlock(self):
        with tf.device('/GPU:0'):
            LayerNorm1 = tf.keras.layers.LayerNormalization(axis=0,dtype='float32')
            LayerNorm1.build(self.input_shape)
            self.weights_shape.append([LayerNorm1])

        with tf.device('/CPU:0'):
            conv_1 = tf.keras.layers.Conv1D(self.hyper_params['B'],self.hyper_params['L'],padding='SAME',dtype='float32')# FIND WHAT THE VALUES ARE
            conv_1_inp = LayerNorm1.compute_output_shape(self.input_shape)
            conv_1.build(conv_1_inp)
            self.weights_shape.append([tf.shape(x).numpy() for x in conv_1.trainable_weights])

        return conv_1.compute_output_shape(conv_1_inp)

    def getShapesofendBlock(self):
        with tf.device('/CPU:0'):
            prelu1 = tf.keras.layers.PReLU(shared_axes=[1])
            prelu1.build(self.Blocks[-1][-1].skip_shape)
            self.weights_shape.append([tf.shape(prelu1.trainable_weights).numpy()])

            conv1 = tf.keras.layers.Conv1D(filters=self.hyper_params['Sc'],kernel_size = self.hyper_params['L'],padding='SAME')
            conv1.build(prelu1.compute_output_shape(self.Blocks[-1][-1].skip_shape)) # should be (batch,B,L)
            self.weights_shape.append([tf.shape(x).numpy() for x in conv1.trainable_weights])
        return

    def forward_pass(self,input_tensor):
        print(tf.shape(input_tensor))
        layernorm_1 = self.weights_shape[0][0](input_tensor)
        print(tf.shape(layernorm_1))
        conv1x1_1 = tf.nn.conv1d(layernorm_1,self.weights[1][0],stride=1,padding='SAME',)
        print(tf.shape(conv1x1_1))
        conv1x1_1_b = tf.add(conv1x1_1,self.weights[1][1])
        print(tf.shape(conv1x1_1_b))
        input = conv1x1_1_b

        print(tf.shape(input))
        outputs = []
        for i in range(self.width):
            for j in range(self.height):
                skip,output = self.Blocks[i][j].forward_pass_block(input)
                outputs.append(skip)
                print(f'output i:{i}, j{j}' ,tf.shape(output))
                input = output
        afterblocks = tf.math.add_n(outputs)

        prelu = parametric_relu(afterblocks,self.weights[-2])
        conv1x1_2 = tf.nn.conv1d(prelu,self.weights[-1][0],stride=1,padding='SAME',)
        conv1x1_2_b = tf.add(conv1x1_2,self.weights[-1][1])
        sigmoid = tf.nn.sigmoid(conv1x1_2_b)

        running_sum = []
        print('#########################################')
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                print( type(self.weights[i][j]))# ,type(self.weights[i][j][0]))

        print('#### #####################################')
        return sigmoid



#if __name__ == '__main__':

#b1 = SeparationStack([4,40000,1],dilation=1)
s1 = SeparationStack([16,40000,1], 7, 2, hyper_params = None)

embed()
