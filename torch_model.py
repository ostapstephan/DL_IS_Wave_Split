import os

import numpy as np
import soundfile as sf
import tensorflow as tf
import tfrecord
import torch
import pandas as pd

from tqdm import tqdm
from IPython import embed
from torch import nn
from torch.utils import data
from scipy.io import wavfile

TOTAL_VAR = 0
BATCH_SIZE = 48
EPOCHS = 1000
CUR_EPOCH = 0


class Data_set(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,split='train'):
        'Initialization'
        self.split=split
        self.base = '/home/ostap/Documents/DL_IS_Wave_Split'

        if split not in ('train', 'test', 'validate'):
            raise ValueError("split must be either 'train', 'test' or 'validate'")

        if split =='train':         # train
            df_i = pd.read_csv(os.path.join(self.base,'train_shuf_meta.csv')).sample(frac=1)
        elif split =='validate':    # val
            df_i = pd.read_csv(os.path.join(self.base,'val_shuf_meta.csv')).sample(frac=1)
        else:                       # test
            df_i = pd.read_csv(os.path.join(self.base,'test_shuf_meta.csv')).sample(frac=1)

        self.df_i = df_i
        self.generator = self.gen(df_i)

    def re_init(self):
        if self.split =='train':         # train
            df_i = pd.read_csv(os.path.join(self.base,'train_shuf_meta.csv')).sample(frac=1)
        elif self.split =='validate':    # val
            df_i = pd.read_csv(os.path.join(self.base,'val_shuf_meta.csv')).sample(frac=1)
        else:                       # test
            df_i = pd.read_csv(os.path.join(self.base,'test_shuf_meta.csv')).sample(frac=1)

        self.df_i = df_i
        self.generator = self.gen(df_i)


    def gen(self,df_i):
        '''
        Make a generator object to return 1 sample at a time
        input: pandas dataframe
        output: one row
        '''
        #print('in_generator')
        ssd_base= '/home/ostap/Documents/DL_IS_Wave_Split/wav8k_split'
        for r in df_i.iterrows():
            # this will use the dataset on the ssd
            base,name =  os.path.split(r[1]['loc_of_wav8k'])
            r[1]['loc_of_wav8k']= os.path.join(ssd_base,name)
            yield r

    def __len__(self):
        'Denotes the total number of samples'
        #TODO find and get length of tf_records
        return self.df_i.shape[0]//2 # getitiem is called twice in one pass
        '''
        global BATCH_SIZE
        if self.split=='train':
            return 445015//BATCH_SIZE
        else:
            return 55628//BATCH_SIZE
        '''

    def normalize_batch(self,tw):
        '''
        this is used to normalize the TF record so that we have two
        equal volume audio tracks
        [batch_size, sample_len_in_sec*sample_rate]
        '''
        #tw = torch.tensor(batch,dtype=torch.float32)
        ts,tm = torch.std_mean(tw,dim= 1)
        out = ((tw.T -tm)/(2*ts)).T # normalize data to be -1 to 1

        # clip outputs to +-2 cuz we found nans in the dataset
        out[out>2]=2
        out[out<-2]=-2
        out[torch.isnan(out)]=0

        return out

    def __getitem__(self, index):
        'Generates one sample of data'

        f0 = next(iter(self.generator))[1]['loc_of_wav8k']
        f1 = next(iter(self.generator))[1]['loc_of_wav8k']
        sr, w0 = wavfile.read(f0)
        sr, w1 = wavfile.read(f1)

        w_torch = torch.Tensor(np.stack([w0,w1]))
        w = self.normalize_batch(w_torch)

        # div by 2 to keep in range -1:1
        x = (torch.sum(w,axis=0)/2).expand(1,w.shape[1])

        return x, w


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
        conv1 = self.conv_1(input_tensor)
        prelu = self.prelu_1(conv1)
        layernorm1 = self.LayerNorm_1(prelu)
        dconv_1 = self.dconv_1(layernorm1)
        prelu2 = self.prelu_2(dconv_1)
        temp = self.Layernorm_2(prelu2)
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
def permutation_loss_helper(label0, pred):

    label1 = torch.roll(label0,1, dims= 0)

    pred_power = torch.sum(pred**2)/pred.shape[1]
    label0_power = torch.sum(label0**2)/label0.shape[1]
    label1_power = torch.sum(label1**2)/label1.shape[1]

    power_term0 =(torch.log10(pred_power)-torch.log10(label0_power))**2
    power_term1 =(torch.log10(pred_power)-torch.log10(label1_power))**2

    a = torch.mean((label0-pred)**2 ,)+power_term0
    b = torch.mean((label1-pred)**2 ,)+power_term1

    if a < b:
        return a
    else:
        return b

def permutation_loss( label0, pred):
    loss=0
    for b in range(label0.shape[0]):
        loss += permutation_loss_helper(label0[b],pred[b])
    #for i in range(label0.shape[0]): # for everything in batch
    #    for j in range(2): # for speaker 0,1
    #        sf.write(f'mixed/l0_{i}_{j}.wav', label0[i,:,j], 8000, 'PCM_16')
    # sf.write(f'mixed/l1_{i}_{j}.wav', label1[i,:,j], 8000, 'PCM_16')
    return loss/label0.shape[0]


def write_output_to_wav(k,truth,predictions,path_,sample_rate=8000):
    print("in output:")
    print(truth.shape)
    print(predictions.shape)
    print((truth[0][0].cpu().detach().numpy()))

    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            sf.write(f'{path_}/{k}output{i}_{j}.wav', ((predictions[i][j].cpu().detach().numpy())) , sample_rate,'PCM_16')
            sf.write(f'{path_}/{k}truth{i}_{j}.wav',  (truth[i][j].cpu().detach().numpy()), sample_rate,'PCM_16')

def eval_ckpt(path_to_ckpt):

    params = {'batch_size': 2,
              'shuffle': False,
              'num_workers': 1}

    dataset = Data_set(split='train')
    generator = torch.utils.data.DataLoader(dataset, **params)

    device = torch.device("cuda")
    gpus = [0]#,1,2] #run this on all 3 of my gpus

    # Construct our model by instantiating the class defined above
    model = Conv_tas_net(7,2) #i[0].shape[1],7,2
    model = nn.DataParallel(model,device_ids=gpus)
    checkpoint = torch.load(path_to_ckpt)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    counter = 0
    for data, labels in generator:

        data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model( data )
        loss = permutation_loss(y_pred, labels) #TODO THIS IS WRONG
        write_output_to_wav(counter,labels,y_pred,'examples')
        if counter % 20 == 0:
            print(f'Validation epoch {CUR_EPOCH}, Counter:{counter}, Loss: {loss}')
        counter+=1

def resume_from_ckpt(path_to_ckpt):
    global CUR_EPOCH
    global BATCH_SIZE
    global EPOCHS

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False,
              'num_workers': 8}

    val_params = {'batch_size': BATCH_SIZE//4,
              'shuffle': False,
              'num_workers': 8}

    ############################################################################
    # Generators
    # https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    training_set = Data_set(split='train')
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Data_set(split='validate')
    validation_generator = torch.utils.data.DataLoader(validation_set, **val_params)
    ############################################################################

    device = torch.device("cuda")
    gpus = [0,1,2] #run this on all 3 of my gpus

    # load ckpt file
    checkpoint = torch.load(path_to_ckpt)
    # state = {'epoch': epoch , 'model': model.state_dict(),
    #         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}

    CUR_EPOCH = checkpoint['epoch']
    # Construct our model by instantiating the class defined above
    model = Conv_tas_net(7,2) #i[0].shape[1],7,2
    model = nn.DataParallel(model,device_ids=gpus)
    model.load_state_dict(checkpoint['model'])
    model.cuda() #Put the model on gpu, To run on multiple GPUs look here: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
    #model.to(device) not sure if its this or model.cuda so dont hate plz
    #print(model)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = permutation_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=.1)
    optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5, last_epoch=-1)
    scheduler.load_state_dict(checkpoint['scheduler'])

    for CUR_EPOCH in range(EPOCHS):
        counter=0
        flag = True
        for data, labels in tqdm(training_generator):
            #data, labels = data.to('cuda:0'), labels.to('cuda:0')
            data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            # Zero gradients,
            optimizer.zero_grad()
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model( data )

            # perform a backward pass, and update the weights.
            # Compute and print loss
            loss = criterion(y_pred, labels) #TODO THIS IS WRONG
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm= 5, norm_type=2)
            optimizer.step()

            if torch.isnan(loss):
                if flag:
                    embed()

            if counter % 20 == 0:
                print(f'Epoch {CUR_EPOCH}, Counter:{counter}, Loss: {loss}')
            counter+=1
        break

        if CUR_EPOCH%1==0 :
            state = {'epoch': CUR_EPOCH , 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(state, f'/share/audiobooks/model_checkpoints/epoch_{CUR_EPOCH}_{loss}.ckpt')

        scheduler.step()

        print('ABOUT TO RE INIT GENERATOR')
        training_set.re_init()
        validation_set.re_init()
        print('SUCCESSFULLY RESET')

    pass


def train():

    global CUR_EPOCH
    global BATCH_SIZE
    # Parameters
    params = {'batch_size': BATCH_SIZE,
              'shuffle': False,
              'num_workers': 8}
             # 'pin_memory': True } #fix this bug w pin memory and num workers <1

    val_params = {'batch_size': BATCH_SIZE//4,
              'shuffle': False,
              'num_workers': 8}

    ############################################################################
    # Generators
    # https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    training_set = Data_set(split='train')
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Data_set(split='validate')
    validation_generator = torch.utils.data.DataLoader(validation_set, **val_params)
    ############################################################################

    gpus = [0,1,2] #run this on all 3 of my gpus

    # Construct our model by instantiating the class defined above
    model = Conv_tas_net(7,2) #i[0].shape[1],7,2
    model = nn.DataParallel(model,device_ids=gpus)

    #model.set_device(gpu) #TODO make this 0,1,2
    model.cuda() #Put the model on gpu, To run on multiple GPUs look here: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html

    #print(model)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = permutation_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5, last_epoch=-1)

    for CUR_EPOCH in range(EPOCHS):
        counter=0
        flag = True
        for data, labels in tqdm(training_generator):
            #data, labels = data.to('cuda:0'), labels.to('cuda:0')
            data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            # Zero gradients,
            optimizer.zero_grad()
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model( data )

            # perform a backward pass, and update the weights.
            # Compute and print loss
            loss = criterion(y_pred, labels) #TODO THIS IS WRONG
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm= 5, norm_type=2)
            optimizer.step()

            if torch.isnan(loss):
                if flag:
                    embed()

            if counter % 20 == 0:
                print(f'Epoch {CUR_EPOCH}, Counter:{counter}, Loss: {loss}')
            counter+=1


        '''
        for data, labels in tqdm(validation_generator):
            data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model( data )
            loss = criterion(y_pred, labels) #TODO THIS IS WRONG

            if counter % 20 == 0:
                print(f'Validation epoch {CUR_EPOCH}, Counter:{counter}, Loss: {loss}')
            counter+=1

        '''


        if CUR_EPOCH%1==0 :
            state = {'epoch': CUR_EPOCH , 'model': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(state, f'/share/audiobooks/model_checkpoints/epoch_{CUR_EPOCH}_{loss}.ckpt')

        scheduler.step()

        print('ABOUT TO RE INIT GENERATOR')
        training_set.re_init()
        validation_set.re_init()
        print('SUCCESSFULLY RESET')


if __name__ == '__main__':
    #train()
    eval_ckpt('/share/audiobooks/model_checkpoints/epoch_6_0.18774129450321198.ckpt')
    #resume_from_ckpt( '/share/audiobooks/model_checkpoints/epoch_9_0.34857264161109924.ckpt')
