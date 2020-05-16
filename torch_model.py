import os

import numpy as np
import soundfile as sf
import tensorflow as tf
import torch

from tqdm import tqdm
from IPython import embed
from torch import nn
from torch.utils import data

TOTAL_VAR = 0
BATCH_SIZE = 32
EPOCHS = 1000
CUR_EPOCH = 0

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
    ds = tf.data.TFRecordDataset(files_ds,compression_type='ZLIB',num_parallel_reads=2)
    # load batch size examples
    ds = ds.batch(batch_size)

    # Parse a batch into a dataset of [audio, label] pairs
    ds = ds.map(lambda x: _parse_batch(x, sample_rate, duration))

    # Repeat the training data for n_epochs. Don't repeat test/validate splits.
    # if split == 'train':
        # ds = ds.repeat(n_epochs)
    return ds.prefetch(buffer_size=1)

def mix_audio(audio ):
    global CUR_EPOCH
    return tf.add(audio,tf.roll(audio,1+CUR_EPOCH,axis=0))

def create_label(audio):
    global CUR_EPOCH
    return tf.squeeze(tf.stack([audio, tf.roll(audio,1+CUR_EPOCH,axis=0)],axis=2), axis = -1)

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,split='train'):
        'Initialization'
        self.split=split
        self.train_ds = get_dataset_from_tfrecords(tfrecords_dir='/share/audiobooks/tf_records', split=split, batch_size=BATCH_SIZE)

    def __len__(self):
        'Denotes the total number of samples'
        #TODO find and get length of tf_records
        global BATCH_SIZE
        if self.split=='train':
            return 445015//BATCH_SIZE
        else:
            return 55628//BATCH_SIZE

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        raw = next(iter(self.train_ds))[0]

        x = mix_audio(tf.expand_dims(raw,2))
        x_train = torch.tensor(x.numpy()).permute(0,2,1)

        y = create_label(tf.expand_dims(raw,2))
        y_train = torch.tensor(y.numpy()).permute(0,2,1)

        return x_train, y_train


class Encoder(torch.nn.Module):
    def __init__(self,hyper_params=False):
        super(Encoder, self).__init__()
        if hyper_params == False:
            self.hyper_params= {
            'N' : 128,
            'L' : 40000//1000,
            'B' : 128,
            'H' : 256,
            'Sc': 128,
            'P' : 3,
            'X' : 7,
            'R' : 2
            }
        else:
            self.hyper_params = hyper_params

        self.padding = (self.hyper_params['L']-1)//2
        self.dconv = nn.Conv1d(1,out_channels=self.hyper_params['H'],kernel_size = self.hyper_params['L'], stride=self.hyper_params['L']//2, padding = self.padding, bias = False)

    def forward(self, input_tensor):
        return self.dconv(input_tensor)

class Decoder(torch.nn.Module):
    def __init__(self,hyper_params=False):
        super(Decoder, self).__init__()
        if hyper_params== False:
            self.hyper_params= {
            'N' : 128,
            'L' : 40000//1000,
            'B' : 128, #out_channels
            'H' : 256,
            'Sc': 128, #sc channels
            'P' : 3,
            'X' : 7,
            'R' : 2
            }
        else:
            self.hyper_params= hyper_params
        self.conv_transpose_1 = nn.ConvTranspose1d(self.hyper_params['H'], out_channels=2, kernel_size = self.hyper_params['L'], stride=(self.hyper_params['L'])//2, padding =10, bias = False)
        #nn.ConvTranspose1d(512, 1, kernel_size=(32,), stride=(16,), bias=False)

    def forward(self, input_tensor):
        return self.conv_transpose_1(input_tensor)


class Conv_1D_Block(torch.nn.Module):
    def __init__(self, dilation= 1, hyper_params = True, B = 128):
        super(Conv_1D_Block, self).__init__()
        if isinstance(hyper_params,type(False)):
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
        else:
            self.hyper_params = hyper_params

        self.hyper_params['B'] = B
        self.dilation = dilation
        self.padding = (dilation)*2
        self.skip_shape = self.hyper_params['Sc']
        self.output_shape = self.hyper_params['B']

        self.conv_1 = nn.Conv1d(in_channels=self.hyper_params['B'],out_channels=self.hyper_params['H'],kernel_size = self.hyper_params['L'],stride=1 )#TODO fix padding
        self.prelu_1 = nn.PReLU(num_parameters=1)
        self.LayerNorm_1 = nn.GroupNorm(self.hyper_params['H'],self.hyper_params['H'])
        self.dconv_1 = nn.Conv1d(in_channels = self.hyper_params['H'],out_channels = self.hyper_params['B'],kernel_size=self.hyper_params['P'],padding=self.dilation,dilation=self.dilation,groups=self.hyper_params['B'])
        self.prelu_2 = nn.PReLU(num_parameters=1)
        self.Layernorm_2 = nn.GroupNorm(self.hyper_params['B'],self.hyper_params['B'])

        self.conv_2_skip = nn.Conv1d(in_channels=self.hyper_params['B'],out_channels=self.hyper_params['B'],kernel_size = self.hyper_params['L'],stride=1 )
        self.conv_2_out =  nn.Conv1d(in_channels=self.hyper_params['B'],out_channels=self.hyper_params['B'],kernel_size = self.hyper_params['L'],stride=1 )

    def forward(self, input_tensor):

        'Forward pass of the conv 1D block'
        #print('inp\t',input_tensor.shape)
        conv1 = self.conv_1(input_tensor)
        #print('conv1\t',conv1.shape)
        prelu = self.prelu_1(conv1)
        #print('prelu\t',prelu.shape)
        layernorm1 = self.LayerNorm_1(prelu)
        #print('l norm1\t',layernorm1.shape)
        dconv_1 = self.dconv_1(layernorm1)
        #print('dconv_1\t',dconv_1.shape)
        prelu2 = self.prelu_2(dconv_1)
        #print('prelu2\t',prelu2.shape)
        temp = self.Layernorm_2(prelu2)
        #print('temp\t',temp.shape)
        return self.conv_2_out(temp),self.conv_2_skip(temp)


class SeparationStack(torch.nn.Module):
    def __init__(self,h=7,w=2,hyper_params=False):
        super(SeparationStack, self).__init__()
        if isinstance(hyper_params, type(True) ):
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
        else:
            self.hyper_params = hyper_params

        self.w = w
        self.h = h
        self.skip_shape = self.hyper_params['Sc']
        self.output_shape = self.hyper_params['B']

        self.LayerNorm_1 = nn.GroupNorm(self.hyper_params['H'],self.hyper_params['H'])
        self.conv_1x1_s = nn.Conv1d(in_channels=self.hyper_params['H'],out_channels=self.hyper_params['B'],kernel_size = self.hyper_params['L'],stride=1,padding=0 )

        self.blocks= nn.ModuleList([])
        for i in range(self.w):
            dil_factor = 1
            for i in range (self.h):
                self.blocks.append(Conv_1D_Block(dilation=dil_factor))
                dil_factor*=2

        self.blocks = self.blocks

        self.prelu_1 = nn.PReLU(1)
        self.conv1x1_e = nn.Conv1d(in_channels=self.hyper_params['B'],out_channels=self.hyper_params['H'],kernel_size = self.hyper_params['L'],stride=1,)

    def forward(self, input_tensor):

        'Forward pass of the conv 1D block'
        ln1 = self.LayerNorm_1(input_tensor)
        conv1s = self.conv_1x1_s(ln1)

        out = conv1s
        skip_accumulator = torch.zeros( out.shape,dtype=torch.float32).cuda()
        for b in self.blocks:
            out,skip= b(out)
            #print(skip.shape,skip_accumulator.shape)
            skip_accumulator+=skip

        return self.conv1x1_e(self.prelu_1(skip_accumulator))

class Conv_tas_net(torch.nn.Module):
    def __init__(self , h,w):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Conv_tas_net, self).__init__()

        self.enc = Encoder()
        self.sep = SeparationStack(h=7, w=2)
        self.dec = Decoder()

    def forward(self, x):
        #print('x.shape',x.shape)
        enc = self.enc(x)
        # block,skip = self.block(enc)
        #print('enc.shape',enc.shape)
        sep = self.sep(enc)

        #print('sep.shape',sep.shape)
        #print('enc.shape',enc.shape)
        dec = self.dec((enc*sep))
        #print(dec.shape)

        return dec

def permutation_loss( label0, pred):
    label1 = torch.roll(label0,1,2)

    a = torch.mean((label0-pred)**2 , dim =[1,2])
    b = torch.mean((label1-pred)**2 , dim =[1,2])

    # comment the saving out for later
    #for i in range(label0.shape[0]): # for everything in batch
    #    for j in range(2): # for speaker 0,1
    #        sf.write(f'mixed/l0_{i}_{j}.wav', label0[i,:,j], 8000, 'PCM_16')
    # sf.write(f'mixed/l1_{i}_{j}.wav', label1[i,:,j], 8000, 'PCM_16')

    aa = torch.mean(a)
    bb = torch.mean(b)

    if aa < bb:
        return aa
    else:
        return bb


def main():
    global CUR_EPOCH
    global BATCH_SIZE
    # Parameters
    params = {'batch_size': 1, #MUST BE 1
              'shuffle': False,
              'num_workers': 0}
             # 'pin_memory': True } #fix this bug w pin memory and num workers <1

    ############################################################################
    # Generators
    # https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    training_set = Dataset(split='train')
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Dataset(split='validate')
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)
    ############################################################################

    #
    # x = mix_audio(tf.expand_dims(i[0],2),0)
    # x_train = torch.tensor(x.numpy()).permute(0,2,1).cuda()

    # Construct our model by instantiating the class defined above
    model = Conv_tas_net(7,2) #i[0].shape[1],7,2
    model.cuda() #Put the model on gpu, To run on multiple GPUs look here: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
    #print(model)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    #criterion = torch.nn.MSELoss(reduction='sum')
    criterion = permutation_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for CUR_EPOCH in range(EPOCHS):
        counter=0
        for data, labels in tqdm(training_generator):
            data, labels = data.squeeze(0), labels.squeeze(0)
            data, labels = data.to('cuda:0'), labels.to('cuda:0')
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model( data )
            # Compute and print loss

            loss = criterion(y_pred, labels) #TODO THIS IS WRONG
            if counter % 1 == 0:
                print(counter, loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if CUR_EPOCH%5==0 :
                torch.save(model.state_dict(), f'/share/audiobooks/model_checkpoints/epoch_{CUR_EPOCH}_{loss.item()}.ckpt')

            counter+=1


if __name__ == '__main__':
    main()
