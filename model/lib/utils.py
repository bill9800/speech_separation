import numpy as np
import librosa
import os 
from tensorflow.python.framework import tensor_util
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
import operator
import itertools
from numpy import inf

#### Load Data ####
# windowing fft/ifft function
def stft(data, fft_size=512, step_size=160,padding=True):
    # short time fourier transform
    if padding == True:
        # for 16K sample rate data, 48192-192 = 48000
        pad = np.zeros(192,)
        data = np.concatenate((data,pad),axis=0)
    # padding hanning window 512-400 = 112
    window = np.concatenate((np.zeros((56,)),np.hanning(fft_size-112),np.zeros((56,))),axis=0)
    win_num = (len(data) - fft_size) // step_size
    out = np.ndarray((win_num, fft_size), dtype=data.dtype)
    for i in range(win_num):
        left = int(i * step_size)
        right = int(left + fft_size)
        out[i] = data[left: right] * window
    F = np.fft.rfft(out, axis=1)
    return F

def istft(F, fft_size=512, step_size=160,padding=True):
    # inverse short time fourier transform
    data = np.fft.irfft(F, axis=-1)
    # padding hanning window 512-400 = 112
    window = np.concatenate((np.zeros((56,)),np.hanning(fft_size-112),np.zeros((56,))),axis=0)
    number_windows = F.shape[0]
    T = np.zeros((number_windows * step_size + fft_size))
    for i in range(number_windows):
        head = int(i * step_size)
        tail = int(head + fft_size)
        T[head:tail] = T[head:tail] + data[i, :] * window
    if padding == True:
        T = T[:48000]
    return T

# combine FFT bins to mel frequency bins
def mel2freq(mel_data,sr,fft_size,n_mel,fmax=8000):
    matrix= librosa.filters.mel(sr, fft_size, n_mel, fmax=fmax)
    return np.dot(mel_data,matrix)

def freq2mel(f_data,sr,fft_size,n_mel,fmax=8000):
    pre_matrix = librosa.filters.mel(sr, fft_size, n_mel, fmax=fmax)
    matrix = pre_matrix.T / np.sum(pre_matrix.T,axis=0)
    return np.dot(f_data,matrix)

# directly time to mel domain transformation
def time_to_mel(data,sr,fft_size,n_mel,step_size,fmax=8000):
    F = stft(data,fft_size,step_size)
    M = freq2mel(F,sr,fft_size,n_mel,fmax=8000)
    return M

def mel_to_time(M,sr,fft_size,n_mel,step_size,fmax=8000):
    F = mel2freq(M,sr,fft_size,n_mel)
    T = istft(F,fft_size,step_size)
    return T

def real_imag_expand(c_data,dim='new'):
    # dim = 'new' or 'same'
    # expand the complex data to 2X data with true real and image number
    if dim == 'new':
        D = np.zeros((c_data.shape[0],c_data.shape[1],2))
        D[:,:,0] = np.real(c_data)
        D[:,:,1] = np.imag(c_data)
        return D
    if dim =='same':
        D = np.zeros((c_data.shape[0],c_data.shape[1]*2))
        D[:,::2] = np.real(c_data)
        D[:,1::2] = np.imag(c_data)
        return D

def real_imag_shrink(F,dim='new'):
    # dim = 'new' or 'same'
    # shrink the complex data to combine real and imag number
    F_shrink = np.zeros((F.shape[0], F.shape[1]))
    if dim =='new':
        F_shrink = F[:,:,0] + F[:,:,1]*1j
    if dim =='same':
        F_shrink = F[:,::2] + F[:,1::2]*1j
    return F_shrink

def power_law(data,power=0.6):
    # assume input has negative value
    mask = np.zeros(data.shape)
    mask[data>=0] = 1
    mask[data<0] = -1
    data = np.power(np.abs(data),power)
    data = data*mask
    return data

def fast_stft(data,power=False,**kwargs):
    # directly transform the wav to the input
    # power law = A**0.3 , to prevent loud audio from overwhelming soft audio
    if power:
        data = power_law(data)
    return real_imag_expand(stft(data))

def fast_istft(F,power=False,**kwargs):
    # directly transform the frequency domain data to time domain data
    # apply power law
    T = istft(real_imag_shrink(F))
    if power:
        T = power_law(T,(1.0/0.6))
    return T


def generate_cRM(Y,S):
    '''

    :param Y: mixed/noisy stft
    :param S: clean stft
    :return: structed cRM
    '''
    M = np.zeros(Y.shape)
    epsilon = 1e-8
    # real part
    M_real = np.multiply(Y[:,:,0],S[:,:,0])+np.multiply(Y[:,:,1],S[:,:,1])
    square_real = np.square(Y[:,:,0])+np.square(Y[:,:,1])
    M_real = np.divide(M_real,square_real+epsilon)
    M[:,:,0] = M_real
    # imaginary part
    M_img = np.multiply(Y[:,:,0],S[:,:,1])-np.multiply(Y[:,:,1],S[:,:,0])
    square_img = np.square(Y[:,:,0])+np.square(Y[:,:,1])
    M_img = np.divide(M_img,square_img+epsilon)
    M[:,:,1] = M_img
    return M

def cRM_tanh_compress(M,K=10,C=0.1):
    '''
    Recall that the irm takes on vlaues in the range[0,1],compress the cRM with hyperbolic tangent
    :param M: crm (298,257,2)
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    '''

    numerator = 1-np.exp(-C*M)
    numerator[numerator == inf] = 1
    numerator[numerator == -inf] = -1
    denominator = 1+np.exp(-C*M)
    denominator[denominator == inf] = 1
    denominator[denominator == -inf] = -1
    crm = K * np.divide(numerator,denominator)

    return crm

def cRM_tanh_recover(O,K=10,C=0.1):
    '''

    :param O: predicted compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return M : uncompressed crm
    '''

    numerator = K-O
    denominator = K+O
    M = -np.multiply((1.0/C),np.log(np.divide(numerator,denominator)))

    return M


def fast_cRM(Fclean,Fmix,K=10,C=0.1):
    '''

    :param Fmix: mixed/noisy stft
    :param Fclean: clean stft
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    '''
    M = generate_cRM(Fmix,Fclean)
    crm = cRM_tanh_compress(M,K,C)
    return crm

def fast_icRM(Y,crm,K=10,C=0.1):
    '''
    :param Y: mixed/noised stft
    :param crm: DNN output of compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return S: clean stft
    '''
    M = cRM_tanh_recover(crm,K,C)
    S = np.zeros(np.shape(M))
    S[:,:,0] = np.multiply(M[:,:,0],Y[:,:,0])-np.multiply(M[:,:,1],Y[:,:,1])
    S[:,:,1] = np.multiply(M[:,:,0],Y[:,:,1])+np.multiply(M[:,:,1],Y[:,:,0])
    return S

def generate_one_sample(audio_path_list,num_speaker,fix_sr=16000,verbose=0):
    '''
    generate one sample from audios in the list

    :param audio_path_list: list contains path of the wav audio file
    :param num_speaker: specify the task for speech separation
    :param fix_sr: fix sample rate
    :return X_sample , y_sample
    '''

    # check path is exist
    for path in audio_path_list:
        if not os.path.exists(path):
            if verbose == 1:
                print('%s is not exist!'% path)
            return

    # initiate variables
    data_list = []
    sr_list = []
    F_list = []  # STFT list for each sample
    X_sample = np.empty(shape=(298,257,2))
    y_sample = np.empty(shape=(298,257,2,num_speaker))

    # import data
    for i in range(num_speaker):
        data, sr = librosa.load(audio_path_list[i],sr=fix_sr)
        data_list.append(data)
        sr_list.append(sr)

    # create mix audio according to mix rate
    mix_rate = 1.0 / float(num_speaker)
    mix = np.zeros(shape=data_list[0].shape)
    for data in data_list:
        np.add(mix,data*mix_rate)

    # transfrom data via STFT and several preprocessing function
    for i in range(num_speaker):
        F = fast_stft(data_list[i])
        F_list.append(F)
    F_mix = fast_stft(mix)
    X_sample = F_mix

    # create cRM for each speaker and fill into y_sample
    for i in range(num_speaker):
        cRM = np.divide(F_list[i], F_mix, out=np.zeros_like(F_list[i]), where=F_mix!=0)
        y_sample[:,:,:,i] = cRM

    # return values
    if verbose == 1:
        print('shape of X: ',X_sample.shape)
        print('shape of y: ',y_sample.shape)
    return X_sample, y_sample

def generate_dataset(sample_range,repo_path,num_speaker=2,**kwargs):
    '''
    A function to generate dataset
    :param sample_range: range of the sample to create the dataset
    :param repo_path: audio repository
    :param num_speaker: number of speaker to separate
    :return: X_data, y_data
    '''
    audio_path_list = []
    X_data = []
    y_data = []
    num_data = 0
    for i in range(sample_range[0],sample_range[1]):
        path = repo_path + '/trim_audio_train%d.wav'%i
        if os.path.exists(path):
            audio_path_list.append(path)

    combinations = itertools.combinations(audio_path_list,num_speaker)
    for combo in combinations:
        num_data += 1
        X_sample,y_sample = generate_one_sample(combo,num_speaker)
        X_data.append(X_sample)
        y_data.append(y_sample)

    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)

    print('number of the data generated: ',num_data)
    return X_data, y_data

#### normalization function ####
def min_max_norm(x):
    # x should be numpy M*N matrix , normalize the N axis
    return (x-np.min(x,axis=0)) / (np.max(x,axis=0)-np.min(x,axis=0))


#### load pre-trained model ####
def load_graph(graph_path,tensorboard=False,**kwargs):
    '''
    :param graph_filename: the path of the pb file
    :return: tensorflow graph
    '''
    with gfile.FastGFile(graph_path,'rb') as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name="")

    if tensorboard:
        writer = tf.summary.FileWriter("log/")
        writer.add_graph(graph)

    return graph


def inspect_operation(graph_path,output_txt_file):
    '''
    :param graph_path: the path of the pb file
    :param output_txt_file: the path of the txt outputfile for inspect the model
    :return:
    '''
    graph = load_graph(graph_path)
    with tf.Session(graph=graph) as sess:
        operations = sess.graph.get_operations()

    ops_dict = {}
    with open(output_txt_file,'w') as f:
        for i, op in enumerate(operations):
            f.write('---------------------------------------------------------------------------------------------\n')
            f.write("{}: op name = {}, op type = ( {} ), inputs = {}, outputs = {}\n".\
                  format(i, op.name, op.type, ", ".join([x.name for x in op.inputs]), ", ".join([x.name for x in op.outputs])))
            f.write('@input shapes:\n')
            for x in op.inputs:
                f.write("name = {} : {}\n".format(x.name, x.get_shape()))
                f.write('@output shapes:\n')
            for x in op.outputs:
                f.write("name = {} : {}\n".format(x.name, x.get_shape()))
            if op.type in ops_dict:
                ops_dict[op.type] += 1
            else:
                ops_dict[op.type] = 1

                f.write('---------------------------------------------------------------------------------------------\n')
        sorted_ops_count = sorted(ops_dict.items(), key=operator.itemgetter(1))
        print('OPS counts:')
        for i in sorted_ops_count:
            print("{} : {}".format(i[0], i[1]))

def SNR(true_file,pred_file):
    T_true,_ = librosa.load(true_file,sr=16000)
    F_true = fast_stft(T_true)
    T_pred, _ = librosa.load(pred_file,sr=16000)
    F_pred = fast_stft(T_pred)
    F_inter = F_true - F_pred
    P_true = np.sum(np.square(F_true[:,:,0])+np.square(F_true[:,:,1]))
    P_inter = np.sum(np.square(F_inter[:,:,0])+np.square(F_inter[:,:,1]))
    return 10*np.log10(np.divide(P_true,P_inter))

def phase_enhance_pred(mix_STFT,pred_file, mode='STFT'):
    if mode=='wav':
        T_pred, _ = librosa.load(pred_file,sr=16000)
        F_pred = fast_stft(T_pred)
    if mode =='STFT':
        F_pred = pred_file
    M = np.sqrt(np.square(F_pred[:,:,0])+np.square(F_pred[:,:,1]))     #magnitude
    print('shape M:',M.shape)
    P = np.arctan(np.divide(mix_STFT[:,:,0],mix_STFT[:,:,1]))          #phase
    print('shape p:',P.shape)
    F_enhance = np.zeros_like(F_pred)
    print('shape enhance',F_enhance.shape)
    F_enhance[:,:,0] = np.multiply(M,np.cos(P))
    F_enhance[:,:,1] = np.multiply(M,np.sin(P))
    print('shape enhance', F_enhance.shape)
    T_enhance = fast_istft(F_enhance)
    return T_enhance

## test code part
if __name__ == '__main__':
    # check data generative function
    CHECK1 = 0
    CHECK2 = 1
    if CHECK1:
        sample_range = (0,20)
        repo_path = os.path.expanduser('../../data/audio/audio_train')
        X_data,y_data = generate_dataset(sample_range,repo_path,num_speaker=2,verbose=1)
        print('shape of the X data: ',X_data.shape)
        print('shape of the y data: ',y_data.shape)

    if CHECK2:
    # check SNR calculation
        original_audio_path = '../../data/audio/audio_train'
        predict_audio_path1 = '../model_v1/pred_with_enhance'
        predict_audio_path2 = '../model_v1/pred_without_enhance'
        print(SNR(original_audio_path + '/trim_audio_train24.wav', original_audio_path + '/trim_audio_train24.wav'))
        #print(SNR(original_audio_path+'/trim_audio_train340.wav',predict_audio_path1+'/00009-00340-00340.wav'))
        print(SNR(original_audio_path + '/trim_audio_train340.wav', predict_audio_path2 + '/00009-00340-00340.wav'))




































