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

BATCH_SIZE = 2

def hard_shape( layer ,batch=BATCH_SIZE):
    #layer should be 0 for encoder, 1 for sep and 2 for decoder
    # encoder: model.encoder.Block.weights_shape
    shape = [
    [
    [[1,1,256],[256]],
    [1,1,256],
    ['lol',[batch, 40000, 256]],
    [[1,256,256],[256]],
    [1,1,256],
    ['lol',[batch, 40000, 256]],
    [[1,256,128],[128]],
    [[1,256,128],[128]]
    ]
    ,
    # Separation block: model.separation_stack.Blocks[0][0].weights_shape
    [
    [[1,128,256],[256]],
    [1,1,256],
    ['lol',[batch, 40000, 256]],
    [[1,256,256],[256]],
    [1,1,256],
    ['lol',[batch, 40000, 256]],
    [[1,256,128],[128]],
    [[1,256,128],[128]]
    ]
    ,
    # Decoder: model.decoder.Block.weights_shape
    [
    [[1,128, 256],[256]],
    [1,1,256],
    ['lol',[batch, 40000, 256]],
    [[1,256,256],[256]],
    [1,1,256],
    ['lol',[batch, 40000, 256]],
    [[1,256,128],[128]],
    [[1,256,2],[2]]
    ]
    ,

    # 3 sep stack non block layers
    [
    ['lol',[batch, 40000, 128]], #dropout
    [[1,128,128], [128]],
    [1,1,128],
    [[1,128,128],[128]]
    ]
    ]

    return shape[layer]
#################################################

weights = []
#@tf.function
#@profile
def gabariofunction():
    ''' sorry for the gabario youre about to witness'''
    global weights
    encoder_shape = hard_shape(0)
    separation_block = hard_shape(1)
    decoder_shape = hard_shape(2)
    separation_stack_shape = hard_shape(3)




    init = tf.initializers.glorot_uniform()
    #encoder
    weights.append(tf.Variable(init(encoder_shape[0][0]),trainable=True,dtype=tf.float32,name='Enc_Conv1'))
    weights.append(tf.Variable(init(encoder_shape[0][1]),trainable=True,dtype=tf.float32,name='ENC_Conv1_b'))
    weights.append(tf.Variable(init(encoder_shape[1]),trainable=True,dtype=tf.float32,name='ENC_PreLu1'))
    weights.append(tf.Variable(init(encoder_shape[3][0]),trainable=True,dtype=tf.float32,name='Enc_Conv2'))
    weights.append(tf.Variable(init(encoder_shape[3][1]),trainable=True,dtype=tf.float32,name='ENC_Conv2_b'))
    weights.append(tf.Variable(init(encoder_shape[4]),trainable=True,dtype=tf.float32,name='ENC_PreLu2'))
    weights.append(tf.Variable(init(encoder_shape[6][0]),trainable=True,dtype=tf.float32,name='Enc_Conv_skip'))
    weights.append(tf.Variable(init(encoder_shape[6][1]),trainable=True,dtype=tf.float32,name='ENC_Conv_skip_b'))
    weights.append(tf.Variable(init(encoder_shape[7][0]),trainable=True,dtype=tf.float32,name='Enc_Conv_output'))
    weights.append(tf.Variable(init(encoder_shape[7][1]),trainable=True,dtype=tf.float32,name='ENC_Conv_output_b'))



    #separationstack
    #1
    # separation_stack_shape
    weights.append(tf.Variable(init(separation_stack_shape[1][0]),trainable=True,dtype=tf.float32,name='sep_first_conv1x1'))
    weights.append(tf.Variable(init(separation_stack_shape[1][1]),trainable=True,dtype=tf.float32,name='sep_first_conv1x1_b'))

    #sep block 1-14
    for i in range(14):
        weights.append(tf.Variable(init(separation_block[0][0]),trainable=True,dtype=tf.float32,name=f'Block{i}_Conv1'))
        weights.append(tf.Variable(init(separation_block[0][1]),trainable=True,dtype=tf.float32,name=f'Block{i}_Conv1_b'))
        weights.append(tf.Variable(init(separation_block[1]),trainable=True,dtype=tf.float32,name=f'Block{i}_PreLu1'))
        weights.append(tf.Variable(init(separation_block[3][0]),trainable=True,dtype=tf.float32,name=f'Block{i}_Conv2'))
        weights.append(tf.Variable(init(separation_block[3][1]),trainable=True,dtype=tf.float32,name=f'Block{i}_Conv2_b'))
        weights.append(tf.Variable(init(separation_block[4]),trainable=True,dtype=tf.float32,name=f'Block{i}_PreLu2'))
        weights.append(tf.Variable(init(separation_block[6][0]),trainable=True,dtype=tf.float32,name=f'Block{i}_Conv_skip'))
        weights.append(tf.Variable(init(separation_block[6][1]),trainable=True,dtype=tf.float32,name=f'Block{i}_Conv_skip_b'))
        weights.append(tf.Variable(init(separation_block[7][0]),trainable=True,dtype=tf.float32,name=f'Block{i}_Conv_output'))
        weights.append(tf.Variable(init(separation_block[7][1]),trainable=True,dtype=tf.float32,name=f'Block{i}_Conv_output_b'))

    weights.append(tf.Variable(init(separation_stack_shape[2]),trainable=True,dtype=tf.float32,name='sep_prelu'))
    weights.append(tf.Variable(init(separation_stack_shape[3][0]),trainable=True,dtype=tf.float32,name='sep_last_conv1x1'))
    weights.append(tf.Variable(init(separation_stack_shape[3][1]),trainable=True,dtype=tf.float32,name='sep_last_conv1x1_b'))

    # decoder
    weights.append(tf.Variable(init(decoder_shape[0][0]),trainable=True,dtype=tf.float32,name='Dec_Conv1'))
    weights.append(tf.Variable(init(decoder_shape[0][1]),trainable=True,dtype=tf.float32,name='Dec_Conv1_b'))
    weights.append(tf.Variable(init(decoder_shape[1]),trainable=True,dtype=tf.float32,name='Dec_PreLu1'))
    weights.append(tf.Variable(init(decoder_shape[3][0]),trainable=True,dtype=tf.float32,name='Dec_Conv2'))
    weights.append(tf.Variable(init(decoder_shape[3][1]),trainable=True,dtype=tf.float32,name='Dec_Conv2_b'))
    weights.append(tf.Variable(init(decoder_shape[4]),trainable=True,dtype=tf.float32,name='Dec_PreLu2'))
    weights.append(tf.Variable(init(decoder_shape[6][0]),trainable=True,dtype=tf.float32,name='Dec_Conv_skip'))
    weights.append(tf.Variable(init(decoder_shape[6][1]),trainable=True,dtype=tf.float32,name='Dec_Conv_skip_b'))
    weights.append(tf.Variable(init(decoder_shape[7][0]),trainable=True,dtype=tf.float32,name='Dec_Conv_output'))
    weights.append(tf.Variable(init(decoder_shape[7][1]),trainable=True,dtype=tf.float32,name='Dec_Conv_output_b'))


gabariofunction()

#@profile
@tf.function
def big_gabario_forward_pass(input_tensor):

    global weights

    #conv 1d block
    #####################################################################################################################
    #encoder
    enc_conv1x1_1 = tf.nn.conv1d(input_tensor, weights[0],stride=1,padding="SAME",dilations=1)
    enc_conv1x1_1_b = tf.add(enc_conv1x1_1,weights[1])

    enc_prelu1 = parametric_relu(enc_conv1x1_1_b, weights[2])

    enc_LayerNorm1 = tf.nn.dropout(enc_prelu1,rate= .1 )

    enc_deconv1 = tf.nn.conv1d(enc_LayerNorm1, weights[3], stride=1, padding="SAME")
    enc_deconv1_b = tf.add(enc_deconv1,weights[4])

    enc_prelu2 = parametric_relu(enc_deconv1_b, weights[5])

    enc_LayerNorm2 = tf.nn.dropout(enc_prelu2,rate= .1 )

    enc_conv1x1_2_skip = tf.nn.conv1d(enc_LayerNorm2,weights[6],stride=1,padding="SAME") #to skip connection
    enc_conv1x1_2_skip_b = tf.add(enc_conv1x1_2_skip,weights[7])
    enc_conv1x1_3_output = tf.nn.conv1d(enc_LayerNorm2, weights[8],stride=1,padding="SAME")
    enc_conv1x1_3_output_b = tf.add(enc_conv1x1_3_output,weights[9])

    #separation_stack
    sep_LayerNorm1 = tf.nn.dropout(enc_conv1x1_3_output_b,rate= .1 )

    sep_conv1x1_1 = tf.nn.conv1d(sep_LayerNorm1, weights[10],stride=1,padding="SAME",dilations=1)
    sep_conv1x1_1_b = tf.add(sep_conv1x1_1,weights[11]) #TODO add the rest of the biases to the model explicitly


    ##############################################################################################
    ##############################################################################################
    ###################################Seperation Blocks##########################################
    ##############################################################################################
    ##############################################################################################

    running_sum = 0
    i=1
    block_0_conv1x1_1 = tf.nn.conv1d( sep_conv1x1_1_b , weights[12],stride=1,padding="SAME",dilations= i)
    block_0_conv1x1_1_b = tf.add(block_0_conv1x1_1,weights[13]) #TODO add the rest of the biases to the model explicitly
    del block_0_conv1x1_1

    block_0_prelu1 = parametric_relu(block_0_conv1x1_1_b, weights[14])
    del block_0_conv1x1_1_b

    block_0_LayerNorm1 = tf.nn.dropout(block_0_prelu1,rate= .1 )
    del block_0_prelu1

    block_0_deconv1 = tf.nn.conv1d(block_0_LayerNorm1, weights[15], stride=1, padding="SAME")
    del block_0_LayerNorm1
    block_0_deconv1_b = tf.add(block_0_deconv1,weights[16])
    del block_0_deconv1

    block_0_prelu2 = parametric_relu(block_0_deconv1_b, weights[17])
    del block_0_deconv1_b

    block_0_LayerNorm2 = tf.nn.dropout(block_0_prelu2,rate= .1 )
    del block_0_prelu2

    block_0_conv1x1_2_skip = tf.nn.conv1d(block_0_LayerNorm2,weights[18],stride=1,padding="SAME") #to skip connection
    block_0_conv1x1_2_skip_b = tf.add(block_0_conv1x1_2_skip,weights[19])
    running_sum = block_0_conv1x1_2_skip_b
    del block_0_conv1x1_2_skip
    del block_0_conv1x1_2_skip_b

    block_0_conv1x1_3_output = tf.nn.conv1d(block_0_LayerNorm2, weights[20],stride=1,padding="SAME")
    block_0_conv1x1_3_output_b = tf.add(block_0_conv1x1_3_output,weights[21])
    del block_0_LayerNorm2
    del block_0_conv1x1_3_output

    

    #i*=2
    block_1_conv1x1_1 = tf.nn.conv1d( block_0_conv1x1_3_output_b , weights[22],stride=1,padding="SAME",dilations= i)
    block_1_conv1x1_1_b = tf.add(block_1_conv1x1_1,weights[23]) #TODO add the rest of the biases to the model explicitly
    del block_1_conv1x1_1

    block_1_prelu1 = parametric_relu(block_1_conv1x1_1_b, weights[24])
    del block_1_conv1x1_1_b

    block_1_LayerNorm1 = tf.nn.dropout(block_1_prelu1,rate= .1 )
    del block_1_prelu1

    block_1_deconv1 = tf.nn.conv1d(block_1_LayerNorm1, weights[25], stride=1, padding="SAME")
    del block_1_LayerNorm1
    block_1_deconv1_b = tf.add(block_1_deconv1,weights[26])
    del block_1_deconv1

    block_1_prelu2 = parametric_relu(block_1_deconv1_b, weights[27])
    del block_1_deconv1_b

    block_1_LayerNorm2 = tf.nn.dropout(block_1_prelu2,rate= .1 )
    del block_1_prelu2

    block_1_conv1x1_2_skip = tf.nn.conv1d(block_1_LayerNorm2,weights[28],stride=1,padding="SAME") #to skip connection
    block_1_conv1x1_2_skip_b = tf.add(block_1_conv1x1_2_skip,weights[29])
    running_sum += block_1_conv1x1_2_skip_b
    del block_1_conv1x1_2_skip
    del block_1_conv1x1_2_skip_b

    block_1_conv1x1_3_output = tf.nn.conv1d(block_1_LayerNorm2, weights[30],stride=1,padding="SAME")
    block_1_conv1x1_3_output_b = tf.add(block_1_conv1x1_3_output,weights[31])
    del block_1_LayerNorm2
    del block_1_conv1x1_3_output



    #i*=2
    block_2_conv1x1_1 = tf.nn.conv1d( block_1_conv1x1_3_output_b , weights[32],stride=1,padding="SAME",dilations= i)
    block_2_conv1x1_1_b = tf.add(block_2_conv1x1_1,weights[33]) #TODO add the rest of the biases to the model explicitly
    del block_2_conv1x1_1

    block_2_prelu1 = parametric_relu(block_2_conv1x1_1_b, weights[34])
    del block_2_conv1x1_1_b

    block_2_LayerNorm1 = tf.nn.dropout(block_2_prelu1,rate= .1 )
    del block_2_prelu1

    block_2_deconv1 = tf.nn.conv1d(block_2_LayerNorm1, weights[35], stride=1, padding="SAME")
    del block_2_LayerNorm1
    block_2_deconv1_b = tf.add(block_2_deconv1,weights[36])
    del block_2_deconv1

    block_2_prelu2 = parametric_relu(block_2_deconv1_b, weights[37])
    del block_2_deconv1_b

    block_2_LayerNorm2 = tf.nn.dropout(block_2_prelu2,rate= .1 )
    del block_2_prelu2

    block_2_conv1x1_2_skip = tf.nn.conv1d(block_2_LayerNorm2,weights[38],stride=1,padding="SAME") #to skip connection
    block_2_conv1x1_2_skip_b = tf.add(block_2_conv1x1_2_skip,weights[39])
    running_sum += block_2_conv1x1_2_skip_b
    del block_2_conv1x1_2_skip
    del block_2_conv1x1_2_skip_b

    block_2_conv1x1_3_output = tf.nn.conv1d(block_2_LayerNorm2, weights[40],stride=1,padding="SAME")
    block_2_conv1x1_3_output_b = tf.add(block_2_conv1x1_3_output,weights[41])
    del block_2_LayerNorm2
    del block_2_conv1x1_3_output



    #i*=2
    block_3_conv1x1_1 = tf.nn.conv1d( block_2_conv1x1_3_output_b , weights[42],stride=1,padding="SAME",dilations= i)
    block_3_conv1x1_1_b = tf.add(block_3_conv1x1_1,weights[43]) #TODO add the rest of the biases to the model explicitly
    del block_3_conv1x1_1

    block_3_prelu1 = parametric_relu(block_3_conv1x1_1_b, weights[44])
    del block_3_conv1x1_1_b

    block_3_LayerNorm1 = tf.nn.dropout(block_3_prelu1,rate= .1 )
    del block_3_prelu1

    block_3_deconv1 = tf.nn.conv1d(block_3_LayerNorm1, weights[45], stride=1, padding="SAME")
    del block_3_LayerNorm1
    block_3_deconv1_b = tf.add(block_3_deconv1,weights[46])
    del block_3_deconv1

    block_3_prelu2 = parametric_relu(block_3_deconv1_b, weights[47])
    del block_3_deconv1_b

    block_3_LayerNorm2 = tf.nn.dropout(block_3_prelu2,rate= .1 )
    del block_3_prelu2

    block_3_conv1x1_2_skip = tf.nn.conv1d(block_3_LayerNorm2,weights[48],stride=1,padding="SAME") #to skip connection
    block_3_conv1x1_2_skip_b = tf.add(block_3_conv1x1_2_skip,weights[49])
    running_sum += block_3_conv1x1_2_skip_b
    del block_3_conv1x1_2_skip
    del block_3_conv1x1_2_skip_b

    block_3_conv1x1_3_output = tf.nn.conv1d(block_3_LayerNorm2, weights[50],stride=1,padding="SAME")
    block_3_conv1x1_3_output_b = tf.add(block_3_conv1x1_3_output,weights[51])
    del block_3_LayerNorm2
    del block_3_conv1x1_3_output



    #i*=2
    block_4_conv1x1_1 = tf.nn.conv1d( block_3_conv1x1_3_output_b , weights[52],stride=1,padding="SAME",dilations= i)
    block_4_conv1x1_1_b = tf.add(block_4_conv1x1_1,weights[53]) #TODO add the rest of the biases to the model explicitly
    del block_4_conv1x1_1

    block_4_prelu1 = parametric_relu(block_4_conv1x1_1_b, weights[54])
    del block_4_conv1x1_1_b

    block_4_LayerNorm1 = tf.nn.dropout(block_4_prelu1,rate= .1 )
    del block_4_prelu1

    block_4_deconv1 = tf.nn.conv1d(block_4_LayerNorm1, weights[55], stride=1, padding="SAME")
    del block_4_LayerNorm1
    block_4_deconv1_b = tf.add(block_4_deconv1,weights[56])
    del block_4_deconv1

    block_4_prelu2 = parametric_relu(block_4_deconv1_b, weights[57])
    del block_4_deconv1_b

    block_4_LayerNorm2 = tf.nn.dropout(block_4_prelu2,rate= .1 )
    del block_4_prelu2

    block_4_conv1x1_2_skip = tf.nn.conv1d(block_4_LayerNorm2,weights[58],stride=1,padding="SAME") #to skip connection
    block_4_conv1x1_2_skip_b = tf.add(block_4_conv1x1_2_skip,weights[59])
    running_sum += block_4_conv1x1_2_skip_b
    del block_4_conv1x1_2_skip
    del block_4_conv1x1_2_skip_b

    block_4_conv1x1_3_output = tf.nn.conv1d(block_4_LayerNorm2, weights[60],stride=1,padding="SAME")
    block_4_conv1x1_3_output_b = tf.add(block_4_conv1x1_3_output,weights[61])
    del block_4_LayerNorm2
    del block_4_conv1x1_3_output



    #i*=2
    block_5_conv1x1_1 = tf.nn.conv1d( block_4_conv1x1_3_output_b , weights[62],stride=1,padding="SAME",dilations= i)
    block_5_conv1x1_1_b = tf.add(block_5_conv1x1_1,weights[63]) #TODO add the rest of the biases to the model explicitly
    del block_5_conv1x1_1

    block_5_prelu1 = parametric_relu(block_5_conv1x1_1_b, weights[64])
    del block_5_conv1x1_1_b

    block_5_LayerNorm1 = tf.nn.dropout(block_5_prelu1,rate= .1 )
    del block_5_prelu1

    block_5_deconv1 = tf.nn.conv1d(block_5_LayerNorm1, weights[65], stride=1, padding="SAME")
    del block_5_LayerNorm1
    block_5_deconv1_b = tf.add(block_5_deconv1,weights[66])
    del block_5_deconv1

    block_5_prelu2 = parametric_relu(block_5_deconv1_b, weights[67])
    del block_5_deconv1_b

    block_5_LayerNorm2 = tf.nn.dropout(block_5_prelu2,rate= .1 )
    del block_5_prelu2

    block_5_conv1x1_2_skip = tf.nn.conv1d(block_5_LayerNorm2,weights[68],stride=1,padding="SAME") #to skip connection
    block_5_conv1x1_2_skip_b = tf.add(block_5_conv1x1_2_skip,weights[69])
    running_sum += block_5_conv1x1_2_skip_b
    del block_5_conv1x1_2_skip
    del block_5_conv1x1_2_skip_b

    block_5_conv1x1_3_output = tf.nn.conv1d(block_5_LayerNorm2, weights[70],stride=1,padding="SAME")
    block_5_conv1x1_3_output_b = tf.add(block_5_conv1x1_3_output,weights[71])
    del block_5_LayerNorm2
    del block_5_conv1x1_3_output



    #i*=2
    block_6_conv1x1_1 = tf.nn.conv1d( block_5_conv1x1_3_output_b , weights[72],stride=1,padding="SAME",dilations= i)
    block_6_conv1x1_1_b = tf.add(block_6_conv1x1_1,weights[73]) #TODO add the rest of the biases to the model explicitly
    del block_6_conv1x1_1

    block_6_prelu1 = parametric_relu(block_6_conv1x1_1_b, weights[74])
    del block_6_conv1x1_1_b

    block_6_LayerNorm1 = tf.nn.dropout(block_6_prelu1,rate= .1 )
    del block_6_prelu1

    block_6_deconv1 = tf.nn.conv1d(block_6_LayerNorm1, weights[75], stride=1, padding="SAME")
    del block_6_LayerNorm1
    block_6_deconv1_b = tf.add(block_6_deconv1,weights[76])
    del block_6_deconv1

    block_6_prelu2 = parametric_relu(block_6_deconv1_b, weights[77])
    del block_6_deconv1_b

    block_6_LayerNorm2 = tf.nn.dropout(block_6_prelu2,rate= .1 )
    del block_6_prelu2

    block_6_conv1x1_2_skip = tf.nn.conv1d(block_6_LayerNorm2,weights[78],stride=1,padding="SAME") #to skip connection
    block_6_conv1x1_2_skip_b = tf.add(block_6_conv1x1_2_skip,weights[79])
    running_sum += block_6_conv1x1_2_skip_b
    del block_6_conv1x1_2_skip
    del block_6_conv1x1_2_skip_b

    block_6_conv1x1_3_output = tf.nn.conv1d(block_6_LayerNorm2, weights[80],stride=1,padding="SAME")
    block_6_conv1x1_3_output_b = tf.add(block_6_conv1x1_3_output,weights[81])
    del block_6_LayerNorm2
    del block_6_conv1x1_3_output



    #i=1
    block_7_conv1x1_1 = tf.nn.conv1d( block_6_conv1x1_3_output_b , weights[82],stride=1,padding="SAME",dilations= i)
    block_7_conv1x1_1_b = tf.add(block_7_conv1x1_1,weights[83]) #TODO add the rest of the biases to the model explicitly
    del block_7_conv1x1_1

    block_7_prelu1 = parametric_relu(block_7_conv1x1_1_b, weights[84])
    del block_7_conv1x1_1_b

    block_7_LayerNorm1 = tf.nn.dropout(block_7_prelu1,rate= .1 )
    del block_7_prelu1

    block_7_deconv1 = tf.nn.conv1d(block_7_LayerNorm1, weights[85], stride=1, padding="SAME")
    del block_7_LayerNorm1
    block_7_deconv1_b = tf.add(block_7_deconv1,weights[86])
    del block_7_deconv1

    block_7_prelu2 = parametric_relu(block_7_deconv1_b, weights[87])
    del block_7_deconv1_b

    block_7_LayerNorm2 = tf.nn.dropout(block_7_prelu2,rate= .1 )
    del block_7_prelu2

    block_7_conv1x1_2_skip = tf.nn.conv1d(block_7_LayerNorm2,weights[88],stride=1,padding="SAME") #to skip connection
    block_7_conv1x1_2_skip_b = tf.add(block_7_conv1x1_2_skip,weights[89])
    running_sum += block_7_conv1x1_2_skip_b
    del block_7_conv1x1_2_skip
    del block_7_conv1x1_2_skip_b

    block_7_conv1x1_3_output = tf.nn.conv1d(block_7_LayerNorm2, weights[90],stride=1,padding="SAME")
    block_7_conv1x1_3_output_b = tf.add(block_7_conv1x1_3_output,weights[91])
    del block_7_LayerNorm2
    del block_7_conv1x1_3_output



    #i*=2
    block_8_conv1x1_1 = tf.nn.conv1d( block_7_conv1x1_3_output_b , weights[92],stride=1,padding="SAME",dilations= i)
    block_8_conv1x1_1_b = tf.add(block_8_conv1x1_1,weights[93]) #TODO add the rest of the biases to the model explicitly
    del block_8_conv1x1_1

    block_8_prelu1 = parametric_relu(block_8_conv1x1_1_b, weights[94])
    del block_8_conv1x1_1_b

    block_8_LayerNorm1 = tf.nn.dropout(block_8_prelu1,rate= .1 )
    del block_8_prelu1

    block_8_deconv1 = tf.nn.conv1d(block_8_LayerNorm1, weights[95], stride=1, padding="SAME")
    del block_8_LayerNorm1
    block_8_deconv1_b = tf.add(block_8_deconv1,weights[96])
    del block_8_deconv1

    block_8_prelu2 = parametric_relu(block_8_deconv1_b, weights[97])
    del block_8_deconv1_b

    block_8_LayerNorm2 = tf.nn.dropout(block_8_prelu2,rate= .1 )
    del block_8_prelu2

    block_8_conv1x1_2_skip = tf.nn.conv1d(block_8_LayerNorm2,weights[98],stride=1,padding="SAME") #to skip connection
    block_8_conv1x1_2_skip_b = tf.add(block_8_conv1x1_2_skip,weights[99])
    running_sum += block_8_conv1x1_2_skip_b
    del block_8_conv1x1_2_skip
    del block_8_conv1x1_2_skip_b

    block_8_conv1x1_3_output = tf.nn.conv1d(block_8_LayerNorm2, weights[100],stride=1,padding="SAME")
    block_8_conv1x1_3_output_b = tf.add(block_8_conv1x1_3_output,weights[101])
    del block_8_LayerNorm2
    del block_8_conv1x1_3_output


#
    #i*=2
    block_9_conv1x1_1 = tf.nn.conv1d( block_8_conv1x1_3_output_b , weights[102],stride=1,padding="SAME",dilations= i)
    block_9_conv1x1_1_b = tf.add(block_9_conv1x1_1,weights[103]) #TODO add the rest of the biases to the model explicitly
    del block_9_conv1x1_1

    block_9_prelu1 = parametric_relu(block_9_conv1x1_1_b, weights[104])
    del block_9_conv1x1_1_b

    block_9_LayerNorm1 = tf.nn.dropout(block_9_prelu1,rate= .1 )
    del block_9_prelu1

    block_9_deconv1 = tf.nn.conv1d(block_9_LayerNorm1, weights[105], stride=1, padding="SAME")
    del block_9_LayerNorm1
    block_9_deconv1_b = tf.add(block_9_deconv1,weights[106])
    del block_9_deconv1

    block_9_prelu2 = parametric_relu(block_9_deconv1_b, weights[107])
    del block_9_deconv1_b

    block_9_LayerNorm2 = tf.nn.dropout(block_9_prelu2,rate= .1 )
    del block_9_prelu2

    block_9_conv1x1_2_skip = tf.nn.conv1d(block_9_LayerNorm2,weights[108],stride=1,padding="SAME") #to skip connection
    block_9_conv1x1_2_skip_b = tf.add(block_9_conv1x1_2_skip,weights[109])
    running_sum += block_9_conv1x1_2_skip_b
    del block_9_conv1x1_2_skip
    del block_9_conv1x1_2_skip_b

    block_9_conv1x1_3_output = tf.nn.conv1d(block_9_LayerNorm2, weights[110],stride=1,padding="SAME")
    block_9_conv1x1_3_output_b = tf.add(block_9_conv1x1_3_output,weights[111])
    del block_9_LayerNorm2
    del block_9_conv1x1_3_output



    #i*=2
    block_10_conv1x1_1 = tf.nn.conv1d( block_9_conv1x1_3_output_b , weights[112],stride=1,padding="SAME",dilations= i)
    block_10_conv1x1_1_b = tf.add(block_10_conv1x1_1,weights[113]) #TODO add the rest of the biases to the model explicitly
    del block_10_conv1x1_1

    block_10_prelu1 = parametric_relu(block_10_conv1x1_1_b, weights[114])
    del block_10_conv1x1_1_b

    block_10_LayerNorm1 = tf.nn.dropout(block_10_prelu1,rate= .1 )
    del block_10_prelu1

    block_10_deconv1 = tf.nn.conv1d(block_10_LayerNorm1, weights[115], stride=1, padding="SAME")
    del block_10_LayerNorm1
    block_10_deconv1_b = tf.add(block_10_deconv1,weights[116])
    del block_10_deconv1

    block_10_prelu2 = parametric_relu(block_10_deconv1_b, weights[117])
    del block_10_deconv1_b

    block_10_LayerNorm2 = tf.nn.dropout(block_10_prelu2,rate= .1 )
    del block_10_prelu2

    block_10_conv1x1_2_skip = tf.nn.conv1d(block_10_LayerNorm2,weights[118],stride=1,padding="SAME") #to skip connection
    block_10_conv1x1_2_skip_b = tf.add(block_10_conv1x1_2_skip,weights[119])
    running_sum += block_10_conv1x1_2_skip_b
    del block_10_conv1x1_2_skip
    del block_10_conv1x1_2_skip_b

    block_10_conv1x1_3_output = tf.nn.conv1d(block_10_LayerNorm2, weights[120],stride=1,padding="SAME")
    block_10_conv1x1_3_output_b = tf.add(block_10_conv1x1_3_output,weights[121])
    del block_10_LayerNorm2
    del block_10_conv1x1_3_output



    #i*=2
    block_11_conv1x1_1 = tf.nn.conv1d( block_10_conv1x1_3_output_b , weights[122],stride=1,padding="SAME",dilations= i)
    block_11_conv1x1_1_b = tf.add(block_11_conv1x1_1,weights[123]) #TODO add the rest of the biases to the model explicitly
    del block_11_conv1x1_1

    block_11_prelu1 = parametric_relu(block_11_conv1x1_1_b, weights[124])
    del block_11_conv1x1_1_b

    block_11_LayerNorm1 = tf.nn.dropout(block_11_prelu1,rate= .1 )
    del block_11_prelu1

    block_11_deconv1 = tf.nn.conv1d(block_11_LayerNorm1, weights[125], stride=1, padding="SAME")
    del block_11_LayerNorm1
    block_11_deconv1_b = tf.add(block_11_deconv1,weights[126])
    del block_11_deconv1

    block_11_prelu2 = parametric_relu(block_11_deconv1_b, weights[127])
    del block_11_deconv1_b

    block_11_LayerNorm2 = tf.nn.dropout(block_11_prelu2,rate= .1 )
    del block_11_prelu2

    block_11_conv1x1_2_skip = tf.nn.conv1d(block_11_LayerNorm2,weights[128],stride=1,padding="SAME") #to skip connection
    block_11_conv1x1_2_skip_b = tf.add(block_11_conv1x1_2_skip,weights[129])
    running_sum += block_11_conv1x1_2_skip_b
    del block_11_conv1x1_2_skip
    del block_11_conv1x1_2_skip_b

    block_11_conv1x1_3_output = tf.nn.conv1d(block_11_LayerNorm2, weights[130],stride=1,padding="SAME")
    block_11_conv1x1_3_output_b = tf.add(block_11_conv1x1_3_output,weights[131])
    del block_11_LayerNorm2
    del block_11_conv1x1_3_output



    #i*=2
    block_12_conv1x1_1 = tf.nn.conv1d( block_11_conv1x1_3_output_b , weights[132],stride=1,padding="SAME",dilations= i)
    block_12_conv1x1_1_b = tf.add(block_12_conv1x1_1,weights[133]) #TODO add the rest of the biases to the model explicitly
    del block_12_conv1x1_1

    block_12_prelu1 = parametric_relu(block_12_conv1x1_1_b, weights[134])
    del block_12_conv1x1_1_b

    block_12_LayerNorm1 = tf.nn.dropout(block_12_prelu1,rate= .1 )
    del block_12_prelu1

    block_12_deconv1 = tf.nn.conv1d(block_12_LayerNorm1, weights[135], stride=1, padding="SAME")
    del block_12_LayerNorm1
    block_12_deconv1_b = tf.add(block_12_deconv1,weights[136])
    del block_12_deconv1

    block_12_prelu2 = parametric_relu(block_12_deconv1_b, weights[137])
    del block_12_deconv1_b

    block_12_LayerNorm2 = tf.nn.dropout(block_12_prelu2,rate= .1 )
    del block_12_prelu2

    block_12_conv1x1_2_skip = tf.nn.conv1d(block_12_LayerNorm2,weights[138],stride=1,padding="SAME") #to skip connection
    block_12_conv1x1_2_skip_b = tf.add(block_12_conv1x1_2_skip,weights[139])
    running_sum += block_12_conv1x1_2_skip_b
    del block_12_conv1x1_2_skip
    del block_12_conv1x1_2_skip_b

    block_12_conv1x1_3_output = tf.nn.conv1d(block_12_LayerNorm2, weights[140],stride=1,padding="SAME")
    block_12_conv1x1_3_output_b = tf.add(block_12_conv1x1_3_output,weights[141])
    del block_12_LayerNorm2
    del block_12_conv1x1_3_output



    #i*=2
    block_13_conv1x1_1 = tf.nn.conv1d( block_12_conv1x1_3_output_b , weights[142],stride=1,padding="SAME",dilations= i)
    block_13_conv1x1_1_b = tf.add(block_13_conv1x1_1,weights[143]) #TODO add the rest of the biases to the model explicitly
    del block_13_conv1x1_1

    block_13_prelu1 = parametric_relu(block_13_conv1x1_1_b, weights[144])
    del block_13_conv1x1_1_b

    block_13_LayerNorm1 = tf.nn.dropout(block_13_prelu1,rate= .1 )
    del block_13_prelu1

    block_13_deconv1 = tf.nn.conv1d(block_13_LayerNorm1, weights[145], stride=1, padding="SAME")
    del block_13_LayerNorm1
    block_13_deconv1_b = tf.add(block_13_deconv1,weights[146])
    del block_13_deconv1

    block_13_prelu2 = parametric_relu(block_13_deconv1_b, weights[147])
    del block_13_deconv1_b

    block_13_LayerNorm2 = tf.nn.dropout(block_13_prelu2,rate= .1 )
    del block_13_prelu2

    block_13_conv1x1_2_skip = tf.nn.conv1d(block_13_LayerNorm2,weights[148],stride=1,padding="SAME") #to skip connection
    block_13_conv1x1_2_skip_b = tf.add(block_13_conv1x1_2_skip,weights[149])
    running_sum += block_13_conv1x1_2_skip_b
    del block_13_conv1x1_2_skip
    del block_13_conv1x1_2_skip_b

    block_13_conv1x1_3_output = tf.nn.conv1d(block_13_LayerNorm2, weights[150],stride=1,padding="SAME")
    block_13_conv1x1_3_output_b = tf.add(block_13_conv1x1_3_output,weights[151])
    del block_13_LayerNorm2
    del block_13_conv1x1_3_output




    sep_prelu1 =  parametric_relu(running_sum, weights[152])

    sep_conv1x1_2 = tf.nn.conv1d(sep_prelu1, weights[153],stride=1,padding="SAME",dilations=1)
    del sep_prelu1
    sep_conv1x1_2_b = tf.add(sep_conv1x1_2,weights[154])
    del sep_conv1x1_2

    sigmoid = tf.sigmoid(sep_conv1x1_2_b)
    del sep_conv1x1_2_b


    #decoder
    dec_conv1x1_1 = tf.nn.conv1d(sigmoid, weights[155],stride=1,padding="SAME",dilations=1)
    del sigmoid
    dec_conv1x1_1_b = tf.add(dec_conv1x1_1,weights[156])
    del dec_conv1x1_1

    dec_prelu1 = parametric_relu(dec_conv1x1_1_b, weights[157])
    del dec_conv1x1_1_b

    dec_LayerNorm1 = tf.nn.dropout(dec_prelu1,rate= .1 )
    del dec_prelu1

    dec_deconv1 = tf.nn.conv1d(dec_LayerNorm1, weights[158], stride=1, padding="SAME")
    del dec_LayerNorm1
    dec_deconv1_b = tf.add(dec_deconv1,weights[159])
    del dec_deconv1

    dec_prelu2 = parametric_relu(dec_deconv1_b, weights[160])
    del dec_deconv1_b

    dec_LayerNorm2 = tf.nn.dropout(dec_prelu2,rate= .1 )
    del dec_prelu2

    dec_conv1x1_2_skip = tf.nn.conv1d(dec_LayerNorm2,weights[161],stride=1,padding="SAME") #to skip connection
    dec_conv1x1_2_skip_b = tf.add(dec_conv1x1_2_skip,weights[162])
    del dec_conv1x1_2_skip
    del dec_conv1x1_2_skip_b


    dec_conv1x1_3_output = tf.nn.conv1d(dec_LayerNorm2, weights[163],stride=1,padding="SAME")
    del dec_LayerNorm2
    return tf.add(dec_conv1x1_3_output,weights[164])


    #####################################################################################################################



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
    ds = tf.data.TFRecordDataset(files_ds,compression_type='ZLIB')#,num_parallel_reads=8)
    del files_ds
    # load batch size examples
    ds = ds.batch(batch_size)

    # Parse a batch into a dataset of [audio, label] pairs
    ds = ds.map(lambda x: _parse_batch(x, sample_rate, duration))

    # Repeat the training data for n_epochs. Don't repeat test/validate splits.
    # if split == 'train':
        # ds = ds.repeat(n_epochs)
    return ds.prefetch(8)

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

def mix_audio(audio, epoch):
    # record = shape[data,label]
    # data = shape[batch_size,40000]
    return tf.add(audio,tf.roll(audio,1+epoch,axis=0))


learning_rate = 0.01
optimizer = tf.optimizers.Adam( learning_rate )

#@tf.function
def loss(target, pred, epoch):
    #label0 = tf.squeeze(tf.stack([target, tf.roll(target,1+epoch,axis=0)],axis=2), axis = -1)
    #lablel1 = label[:,:,0],label[:,:,1]
    #label1= tf.Variable([label0[:,:,1],label0[:,:,0]])
    #label1 =tf.transpose(label1,[1,2,0])

    #a = tf.math.reduce_mean((label0-pred)**2 , axis =[1,2])
    #b = tf.math.reduce_mean((label1-pred)**2 , axis =[1,2])

        # comment the saving out for later
        #for i in range(label0.shape[0]): # for everything in batch
        #    for j in range(2): # for speaker 0,1
        #        sf.write(f'mixed/l0_{i}_{j}.wav', label0[i,:,j], 8000, 'PCM_16')
        # sf.write(f'mixed/l1_{i}_{j}.wav', label1[i,:,j], 8000, 'PCM_16')

    #l0 = tf.losses.mean_squared_error(label0, pred )
    #l1 = tf.losses.mean_squared_error(label1, pred )

    #a = tf.math.reduce_mean(l0)
    #b = tf.math.reduce_mean(l1)

    #if a < b:
    #    return l0
            # idk why but this is shape [batch,40000] not [batch,] or
            # [batch,40000,2] so i reduced sum on batch
    #else:
    ##    return l1
    return tf.losses.mean_squared_error(tf.squeeze(tf.stack([target, tf.roll(target,1+epoch,axis=0)],axis=2), axis = -1),pred)

# @tf.function
# def forward_pass(self,x):
#     # forward_pass
#     mixed = mix_audio(x,self.epoch)
#     return big_gabario_forward_pass(mixed)
#@tf.function
def train_step( inputs, epoch ):
    global weights
    global optimizer
    with tf.GradientTape() as tape:
    #embed()
        #print('train step\t', inputs.shape)
    #big_gabario_forward_pass(mix_audio(inputs,epoch))
        current_loss = loss(  inputs, big_gabario_forward_pass( mix_audio(inputs,epoch)),epoch)
        #embed()
        print("I AM HERE")
    #grads = tape.gradient(current_loss, weights )
    optimizer.apply_gradients( zip(tape.gradient( current_loss, weights ) , weights ) )

    # #print( tf.reduce_mean( current_loss ).numpy() )

#@tf.function
def fit(dataset,num_epochs):
    for epoch in range( num_epochs ):
        # i = 0
        for features in dataset:
            # i+=1
            audio, label = tf.expand_dims(features[0],2) , features[1]
            #big_gabario_forward_pass(mix_audio(audio,epoch))
            train_step(audio,epoch)
            break
        break
            # if i == 50 :
                # break

def main():
    # Train
    tf.config.set_soft_device_placement(True) #runs on cpu if gpu isn't availible
    train_ds = get_dataset_from_tfrecords(tfrecords_dir='/home/car-sable/libre_data/tf_records',batch_size=BATCH_SIZE)
    # 22 on gpu 0 is max rn
    # 12 on gpu 1

    # this is so we get the input shape of the dataset
    for i in train_ds:
        #model = Model(tf.shape(tf.expand_dims(i[0],2)),7,2)
        break

    print('first',tf.expand_dims(i[0],2))
    #cost = model.loss(tf.expand_dims(i[0],2), model.forward_pass(tf.expand_dims(i[0],2))
    #print('cost',cost)
    # embed()
    #model.forward_pass(tf.expand_dims(i[0],2))
    #model.train_step( tf.expand_dims(i[0],2) )
    fit(train_ds,num_epochs=10)
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
    print('Total Var',TOTAL_VAR)
