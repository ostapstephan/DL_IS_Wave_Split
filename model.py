import os
import tensorflow as tf
import numpy as np
#import scipy.io.wavfile
from tqdm import tqdm
import gc
from IPython import embed

AUTOTUNE = tf.data.experimental.AUTOTUNE

import soundfile as sf
TOTAL_VAR = 0

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
    del files_ds
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
    # model.fit(train_ds, epochs=10)
    # model.fit(train_ds, epochs=10)
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

class Conv_1D_Block(object):
    def __init__(self,input_shape,dilation=1, B = 128):
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
        self.hyper_params['B'] = B

        self.weights_shape = []
        self.input_shape=input_shape
        self.dilation = dilation
        self.initializer = tf.initializers.glorot_uniform()
        self.skip_shape = []
        self.output_shape = []
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


        LayerNorm1 = tf.keras.layers.LayerNormalization(axis=2,dtype='float32')
        LayerNorm1_inp = prelu1.compute_output_shape(prelu1_inp)
        LayerNorm1.build(LayerNorm1_inp)
        self.weights_shape.append(['TEMP',LayerNorm1_inp]) # append the whole layer

        dconv1 = tf.keras.layers.Conv1D(filters=self.hyper_params['H'], kernel_size = self.hyper_params['L'], padding = 'same') #This should be a depthwise conv but they didnt do it so neither will we
        dconv1_inp = LayerNorm1.compute_output_shape(LayerNorm1_inp)
        dconv1.build(dconv1_inp)
        self.weights_shape.append([tf.shape(x).numpy() for x in dconv1.trainable_weights])

        prelu2 = tf.keras.layers.PReLU(shared_axes=[1])
        prelu2_inp = dconv1.compute_output_shape(dconv1_inp)
        prelu2.build(prelu2_inp)
        self.weights_shape.append([tf.shape(prelu2.trainable_weights).numpy()])

        LayerNorm2 = tf.keras.layers.LayerNormalization(axis=2,dtype='float32')
        LayerNorm2_inp = prelu2.compute_output_shape(prelu2_inp)
        LayerNorm2.build(LayerNorm2_inp)
        self.weights_shape.append(['TEMP',LayerNorm2_inp]) # appending the whole layer

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

        tf.keras.backend.clear_session()

        gc.collect()
        del conv1
        del prelu1
        del prelu2
        del dconv1_inp

        return

    def get_weights(self):
        global TOTAL_VAR
        for i in range(len(self.weights_shape)): # weight_shape_list
            if not self.weights_shape[i][0] == 'TEMP':
                a = [tf.Variable(self.initializer(w),trainable=True,dtype=tf.float32) for w in self.weights_shape[i] ]
                self.weights.append( a   )
                TOTAL_VAR += sum([tf.reduce_prod(tf.shape(x)).numpy() for x in a ])
                ##print( [(tf.shape(x)).numpy() for x in a] )
            else:
                temp =  tf.keras.layers.LayerNormalization(axis=2,dtype='float32')
                temp.build(self.weights_shape[i][1])
                a = [temp.trainable_weights]
                self.weights_shape[i] = [temp]
                self.weights.append(a)
                TOTAL_VAR += sum([tf.reduce_prod(tf.shape(x)).numpy() for x in a ])
                ##print( [(tf.shape(x)).numpy() for x in a] )
            tf.keras.backend.clear_session()

    def section_1(self,input_tensor):
        conv1x1_1 = tf.nn.conv1d(input_tensor,self.weights[0][0],stride=1,padding="SAME",dilations=self.dilation)
        conv1x1_1_b = tf.add(conv1x1_1,self.weights[0][1]) #TODO add the rest of the biases to the model explicitly
        del conv1x1_1
        ##print(tf.shape(conv1x1_1_b),'conv_1d')
        prelu1 = parametric_relu(conv1x1_1_b, self.weights[1])
        del conv1x1_1_b
        return prelu1

    def section_2(self,input_tensor):
        LayerNorm1 = self.weights_shape[2][0](input_tensor)
        return LayerNorm1
        # weights[2] == keras_layers.get_trakinable_weights

    def section_3(self,input_tensor):
        deconv1 = tf.nn.conv1d(input_tensor, self.weights[3][0], stride=1, padding="SAME")

        deconv1_b = tf.add(deconv1,self.weights[3][1])
        del deconv1
        return deconv1_b

    def section_4(self,input_tensor):
        LayerNorm2 = self.weights_shape[5][0](input_tensor) # weights[5] == keras_layers.get_trainable_weights
        return LayerNorm2


    def section_5(self,input_tensor):
        conv1x1_2_skip = tf.nn.conv1d(input_tensor,self.weights[6][0],stride=1,padding="SAME") #to skip connection
        conv1x1_2_skip_b = tf.add(conv1x1_2_skip,self.weights[6][1])
        del conv1x1_2_skip

        return conv1x1_2_skip_b

    def section_6(self,input_tensor):
        print('section_6 input , weight[0], weights[1]',input_tensor.shape,self.weights[7][0].shape,self.weights[7][1].shape)
        conv1x1_3_output = tf.nn.conv1d(input_tensor,self.weights[7][0],stride=1,padding="SAME")
        conv1x1_3_output_b = tf.add(conv1x1_3_output,self.weights[7][1])
        del conv1x1_3_output
        return conv1x1_3_output_b

    def forward_pass_block(self,input_tensor):
        # conv1x1_1 = tf.nn.conv1d(input_tensor,self.weights[0][0],stride=1,padding="SAME",dilations=self.dilation)
        # conv1x1_1_b = tf.add(conv1x1_1,self.weights[0][1]) #TODO add the rest of the biases to the model explicitly
        # del conv1x1_1
        # ##print(tf.shape(conv1x1_1_b),'conv_1d')
        # prelu1 = parametric_relu(conv1x1_1_b, self.weights[1])
        # del conv1x1_1_b
        # LayerNorm1 = self.weights_shape[2][0](prelu1)
        # del prelu1
        # # weights[2] == keras_layers.get_trakinable_weights
        # deconv1 = tf.nn.conv1d(LayerNorm1, self.weights[3][0], stride=1, padding="SAME")
        # del LayerNorm1
        # #Should be depthwise seperable convolution not transpose
        # deconv1_b = tf.add(deconv1,self.weights[3][1])
        # del deconv1
        sec1 = self.section_1(input_tensor)
        sec2 = self.section_2(sec1)
        del sec1
        sec3 = self.section_3(sec2)
        del sec2
        sec4 = self.section_4(sec3)
        del sec3
        return self.section_5(sec4), self.section_6(sec4)



        # prelu2 = parametric_relu(deconv1_b, self.weights[4])
        # del deconv1_b
        # LayerNorm2 = self.weights_shape[5][0](prelu2) # weights[5] == keras_layers.get_trainable_weights
        # del prelu2
        # conv1x1_2_skip = tf.nn.conv1d(LayerNorm2,self.weights[6][0],stride=1,padding="SAME") #to skip connection
        # conv1x1_2_skip_b = tf.add(conv1x1_2_skip,self.weights[6][1])
        # del conv1x1_2_skip
        #
        # conv1x1_3_output = tf.nn.conv1d(LayerNorm2,self.weights[7][0],stride=1,padding="SAME")
        # del LayerNorm2
        # conv1x1_3_output_b = tf.add(conv1x1_3_output,self.weights[7][1])
        # del conv1x1_3_output
        # gc.collect()
        # return conv1x1_2_skip_b, conv1x1_3_output_b

        # this is cancer but prevents the del statements
        # conv1x1_1 =  self.weights_shape[5][0]( \
        #             parametric_relu( \
        #                 tf.add( \
        #                     tf.nn.conv1d( \
        #                         self.weights_shape[2][0]( \
        #                             parametric_relu( \
        #                                 tf.add( \
        #                                     tf.nn.conv1d(\
        #                                         input_tensor,self.weights[0][0],stride=1,padding="SAME",dilations=self.dilation),
        #                                     self.weights[0][1])
        #                             ,self.weights[1])),
        #                         self.weights[3][0], stride=1, padding="SAME")
        #                     ,self.weights[3][1]),
        #                 self.weights[4])
        #             )
        #
        #
        #
        # conv1x1_2_skip = tf.add(tf.nn.conv1d(conv1x1_1,self.weights[6][0],stride=1,padding="SAME"),self.weights[6][1]) #to skip connection
        #
        # conv1x1_3_output = tf.add(tf.nn.conv1d(conv1x1_1,self.weights[7][0],stride=1,padding="SAME"), self.weights[7][1])
        # del conv1x1_1
        #
        # return conv1x1_2_skip, conv1x1_3_output


class Coder(object):
    def __init__(self,input_shape,start=True):
        ##print(input_shape,'coder input shape')
        if  start:
            self.Block = Conv_1D_Block(input_shape)
        else:
            self.Block = Conv_1D_Block(input_shape, B = 2)

        self.output_shape = self.Block.output_shape

    def getShapesofBlock(self):
        return self.Block.getShapesofBlock()

    def forward_pass(self,input_tensor):
        _, output = self.Block.forward_pass_block(input_tensor)
        del _
        gc.collect()
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

        self.output_shape= []
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
                add.append(Conv_1D_Block(shape,2**j))
                shape =add[-1].output_shape
            self.Blocks.append(add)
            del add

        self.getShapesofendBlock()
        self.get_weights()


    def get_weights(self):
        global TOTAL_VAR
        for i in range(len(self.weights_shape)): # weight_shape_list
            if not self.weights_shape[i][0] == 'TEMP':
                a =[tf.Variable(self.initializer(w),trainable=True,dtype=tf.float32) for w in self.weights_shape[i] ]
                self.weights.append( a )
                TOTAL_VAR += sum([tf.reduce_prod(tf.shape(x)).numpy() for x in a ])
            else:
                temp =  tf.keras.layers.LayerNormalization(axis=2,dtype='float32')
                temp.build(self.weights_shape[i][1])
                a = [temp.trainable_weights]
                self.weights_shape[i] = [temp]
                self.weights.append(a)
                TOTAL_VAR += sum([tf.reduce_prod(tf.shape(x)).numpy() for x in a ])
            tf.keras.backend.clear_session()

        temp= []
        for i in range(self.width):
            for j in range(self.height):
                temp.append(self.Blocks[i][j].weights)
                tf.keras.backend.clear_session()
        self.weights = self.weights[:-2] + temp + self.weights[-2:]


    def getShapesofbeginningBlock(self):
        LayerNorm1 = tf.keras.layers.LayerNormalization(axis=2,dtype='float32')
        LayerNorm1.build(self.input_shape)
        self.weights_shape.append(['TEMP',self.input_shape])

        conv_1 = tf.keras.layers.Conv1D(self.hyper_params['B'],self.hyper_params['L'],padding='SAME',dtype='float32')# FIND WHAT THE VALUES ARE
        conv_1_inp = LayerNorm1.compute_output_shape(self.input_shape)
        conv_1.build(conv_1_inp)
        self.weights_shape.append([tf.shape(x).numpy() for x in conv_1.trainable_weights])
        shape_out = conv_1.compute_output_shape(conv_1_inp)

        tf.keras.backend.clear_session()
        return shape_out

    def getShapesofendBlock(self):
        prelu1 = tf.keras.layers.PReLU(shared_axes=[1])
        prelu1.build(self.Blocks[-1][-1].skip_shape)
        self.weights_shape.append([tf.shape(prelu1.trainable_weights).numpy()])

        conv1 = tf.keras.layers.Conv1D(filters=self.hyper_params['Sc'],kernel_size = self.hyper_params['L'],padding='SAME')
        conv1.build(prelu1.compute_output_shape(self.Blocks[-1][-1].skip_shape)) # should be (batch,B,L)
        self.weights_shape.append([tf.shape(x).numpy() for x in conv1.trainable_weights])
        self.output_shape = conv1.compute_output_shape(prelu1.compute_output_shape(self.Blocks[-1][-1].skip_shape))

        tf.keras.backend.clear_session()
        return

    def forward_pass(self,input_tensor):
        ##print(tf.shape(input_tensor))
        layernorm_1 = self.weights_shape[0][0](input_tensor)
        ##print(tf.shape(layernorm_1))
        conv1x1_1 = tf.nn.conv1d(layernorm_1,self.weights[1][0],stride=1,padding='SAME',)
        del layernorm_1
        ##print(tf.shape(conv1x1_1))
        conv1x1_1_b = tf.add(conv1x1_1,self.weights[1][1])
        del conv1x1_1
        ##print(tf.shape(conv1x1_1_b))
        input = conv1x1_1_b
        del conv1x1_1_b

        ##print(tf.shape(input))
        #temp_output = 0
        for i in range(self.width):
            for j in range(self.height):
                if (i == 0) and (j == 0):
                    temp_output,output  = self.Blocks[i][j].forward_pass_block(input)
                    input = output
                else:
                    skip,output = self.Blocks[i][j].forward_pass_block(input)
                    temp_output = temp_output+skip
                    ##print(f'output i:{i}, j{j}' ,tf.shape(output))
                    del skip
                    input = output
                    del output

        prelu = parametric_relu(temp_output,self.weights[-2])
        del temp_output
        conv1x1_2 = tf.nn.conv1d(prelu,self.weights[-1][0],stride=1,padding='SAME',)
        del prelu
        conv1x1_2_b = tf.add(conv1x1_2,self.weights[-1][1])
        del conv1x1_2
        sigmoid = tf.nn.sigmoid(conv1x1_2_b)

        #running_sum =[]
        ##print('#########################################')
        #for i in range(len(self.weights)):
        #    for j in range(len(self.weights[i])):
                #print( type(self.weights[i][j]))# ,type(self.weights[i][j][0]))

        #print('#### #####################################')
        gc.collect()
        return sigmoid

class Model(object):
    def __init__(self,input_shape,Height,Width):
        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
        self.epoch = 0
        self.num_epochs = 22
        self.input_shape = input_shape

        #self.shapes = self.get_shape()

        learning_rate = 0.01
        self.optimizer = tf.optimizers.Adam( learning_rate )

        def get_weight( shape , name ):
            return tf.Variable( self.initializer( shape ) , name=name , trainable=True , dtype=tf.float32 )

        self.initializer = tf.initializers.glorot_uniform()
        self.weights = []

        self.weights_training = [] #this will be used to send the weights to adam

        # build the encoder,SeparationStack, and decoder
        self.encoder = Coder(self.input_shape)
        self.separation_stack  = SeparationStack(self.encoder.output_shape, Height, Width)
        self.decoder = Coder(self.separation_stack.output_shape,start=False)

        self.append_weights(self.encoder.Block.weights)
        self.append_weights(self.separation_stack.weights)
        self.append_weights(self.decoder.Block.weights)

    def append_weights(self,weights):
        '''
        Recursively get all the weight tensors from the list of lists
        '''
        #print(type(weights))
        if isinstance(weights,type(tf.Variable([[0]],trainable=True))):
            self.weights_training.append(weights)
            return

        for i in range(len(weights)):
            self.append_weights(weights[i])

    def loss(self, target, pred):
        label0 = tf.squeeze(tf.stack([target, tf.roll(target,1+self.epoch,axis=0)],axis=2), axis = -1)
        #lablel1 = label[:,:,0],label[:,:,1]
        label1= tf.Variable([label0[:,:,1],label0[:,:,0]])
        label1 =tf.transpose(label1,[1,2,0])

        a = tf.math.reduce_mean((label0-pred)**2 , axis =[1,2])
        b = tf.math.reduce_mean((label1-pred)**2 , axis =[1,2])

        # comment the saving out for later
        #for i in range(label0.shape[0]): # for everything in batch
        #    for j in range(2): # for speaker 0,1
        #        sf.write(f'mixed/l0_{i}_{j}.wav', label0[i,:,j], 8000, 'PCM_16')
        # sf.write(f'mixed/l1_{i}_{j}.wav', label1[i,:,j], 8000, 'PCM_16')

        l0 = tf.losses.mean_squared_error(label0, pred )
        l1 = tf.losses.mean_squared_error(label1, pred )

        a = tf.math.reduce_mean(l0)
        b = tf.math.reduce_mean(l1)

        if a < b:
            return l0
            # idk why but this is shape [batch,40000] not [batch,] or
            # [batch,40000,2] so i reduced sum on batch
        else:
            return l1

    def forward_pass(self,x):
        # forward_pass
        mixed = mix_audio(x,self.epoch)
        encoded = self.encoder.forward_pass(mixed)
        del mixed
        sep = self.separation_stack.forward_pass(encoded)
        combined = encoded*sep
        del encoded
        del sep
        decoded = self.decoder.forward_pass(combined)
        del combined
        return decoded

    def train_step( self, inputs ):
        with tf.GradientTape() as tape:
            print('train step\t', inputs.shape)
            current_loss = self.loss(  inputs, self.forward_pass( inputs ) )
            #grads = tape.gradient( current_loss, self.weights_training )
            embed()
            self.optimizer.apply_gradients( zip(tape.gradient( current_loss, self.weights_training ) , self.weights_training ) )
            embed()

        # #print( tf.reduce_mean( current_loss ).numpy() )

    def fit(self, dataset):
        for self.epoch in range( self.num_epochs ):
            # i = 0
            for features in tqdm(dataset):
                # i+=1
                audio, label = tf.expand_dims(features[0],2) , features[1]
                a = self.weights_training[0][:,:,:10]
                self.train_step( audio
                break
            break
                # if i == 50 :
                    # break

def main():
    # Train
    tf.config.set_soft_device_placement(True) #runs on cpu if gpu isn't availible
    train_ds = get_dataset_from_tfrecords(tfrecords_dir='/share/audiobooks/tf_records',batch_size=2)
    # 22 on gpu 0 is max rn
    # 12 on gpu 1

    # this is so we get the input shape of the dataset
    for i in train_ds:
        model = Model(tf.shape(tf.expand_dims(i[0],2)),1,1)
        break

    print('first',tf.expand_dims(i[0],2))
    #cost = model.loss(tf.expand_dims(i[0],2), model.forward_pass(tf.expand_dims(i[0],2)) )

    model.train_step( tf.expand_dims(i[0],2) )
    # model.fit(train_ds)

    # counter = 0
    # for i in train_ds:
    #     cost = model.loss(tf.expand_dims(i[0],2), model.forward_pass(tf.expand_dims(i[0],2)) )
    #     print(counter)
    #     counter+=1
    #
    #     cost = model.loss(tf.expand_dims(i[0],2), model.forward_pass(tf.expand_dims(i[0],2)) )
    #     print(tf.reduce_mean(cost))
    #
    #     break


    #embed()
    #model.fit(train_ds)

    '''
    # Test/ Create Audios
    for record in train_ds:
        #out = model.forward_pass(record[0]).numpy()
        out = model.forward_pass(tf.expand_dims(record[0],2))
        in_ = tf.expand_dims(record[0],2)
        for i in range(out.shape[0]):
            for j in range(2):
                sf.write(f'mixed/gabario{i}{j}.wav', out[i,:,j], 8000, 'PCM_16')
                sf.write(f'mixed/a{i}{j}.wav', out[i,:,j], 8000, 'PCM_16')
                sf.write(f'mixed/b{i}{j}.wav', out[i,:,j], 8000, 'PCM_16')

            #scipy.io.wavfile.write(f'mixed/gabario{i}.wav', 8000, out[1,:,0])
            #scipy.io.wavfile.write(f'mixed/mix{i}.wav', 8000, record[0][i].numpy())
            #scipy.io.wavfile.write(f'mixed/a{i}.wav', 8000, record[0][i][:,:,0].numpy())
            #scipy.io.wavfile.write(f'mixed/b{i}.wav', 8000, record[0][i][:,:,1].numpy())
        break

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
    print(TOTAL_VAR)
