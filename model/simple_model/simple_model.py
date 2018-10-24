# simple model with gain method for separating 2 people speech

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import os
import scipy.io.wavfile as wavfile
# OPTION
TRAIN = 0
TEST = 0
INVERSE_CHECK = 1


# read the data in time domain and its sample rate
path1 = '../../data/audio/audio/trim_audio1.wav'
path2 = '../../data/audio/audio/trim_audio2.wav'
path3 = '../../data/audio/audio/mix_audio.wav'
with open(path1, 'rb') as f:
    data1, sr1 = librosa.load(path1, sr=None)

with open(path2, 'rb') as f:
    data2, sr2 = librosa.load(path2, sr=None)

with open(path3, 'rb') as f:
    mix_data, mix_sr = librosa.load(path3, sr=None)

####################################
# preprocessing #

# parameter
fft_size = 1024
step_size = fft_size // 3
n_mel = 60

# fft transformation

def window_fft(data,fft_size,step_size):
    window = np.hamming(fft_size)
    number_windows = (data.shape[0]-fft_size)//step_size
    output = np.ndarray((number_windows,fft_size),dtype=data.dtype)

    for i in range(number_windows):
        head = int(i*step_size)
        tail = int(head+fft_size)
        output[i] = data[head:tail]*window

    F = np.fft.rfft(output,axis=-1)
    return F

def rev_window_fft(F,fft_size,step_size):
    data = np.fft.irfft(F,axis=-1)
    window = np.hamming(fft_size)
    number_windows = F.shape[0]

    T = np.zeros((number_windows*step_size+fft_size))
    for i in range(number_windows):
        head = int(i*step_size)
        tail = int(head+fft_size)
        T[head:tail] = T[head:tail]+data[i,:]*window
    return T

# mel2freq_matrix generation
def mel2freq_matrix(sr,fft_size,n_mel,fmax=8000):
    matrix= librosa.filters.mel(sr, fft_size, n_mel, fmax=fmax)
    return matrix

def freq2mel_matrix(sr,fft_size,n_mel,fmax=8000):
    pre_matrix = mel2freq_matrix(sr, fft_size, n_mel, fmax)
    matrix = pre_matrix.T / np.sum(pre_matrix.T,axis=0)
    return matrix


# mel transformation
def time_to_mel(data,sr,fft_size,n_mel,step_size,fmax=8000):
    fft_data =  window_fft(data,fft_size,step_size)
    mel_data = np.dot(fft_data,freq2mel_matrix(sr,fft_size,n_mel,fmax))
    return mel_data

def mel_to_time(mel_data,sr,fft_size,n_mel,step_size,fmax=8000):
    F = np.dot(mel_data,mel2freq_matrix(sr,fft_size,n_mel,fmax))
    T = rev_window_fft(F,fft_size,step_size)
    return T

def mel_real_imag_expand(mel_data):
    # expand the mel data to 2X data with true real and image number
    D = np.zeros((mel_data.shape[0],mel_data.shape[1]*2))
    D[:,::2] = np.real(mel_data)
    D[:,1::2] = np.imag(mel_data)
    return D

def rev_gain(gain,max,sr,fft_size,n_mel):
    M_G = np.zeros((gain.shape[0], gain.shape[1] * 2))
    M_G = gain[:,::2]+1j*gain[:,1::2]
    F_G = np.dot(M_G,mel2freq_matrix(sr,fft_size,n_mel))
    return F_G * max

# normalization function
def min_max_norm(x):
    # x should be numpy M*N matrix , normalize the N axis
    return (x-np.min(x,axis=0)) / (np.max(x,axis=0)-np.min(x,axis=0))

########################################
# create data sets

D1 = time_to_mel(data1,sr1,fft_size,n_mel,step_size)
D2 = time_to_mel(data2,sr2,fft_size,n_mel,step_size)
D_mix = time_to_mel(mix_data,mix_sr,fft_size,n_mel,step_size)

G1 = np.abs(mel_real_imag_expand(D1)/mel_real_imag_expand(D_mix))
G2 = np.abs(mel_real_imag_expand(D2)/mel_real_imag_expand(D_mix))

X_train = mel_real_imag_expand(D_mix[:int(D_mix.shape[0]*0.8),:])
G1_train = G1[:int(D_mix.shape[0]*0.8),:]
G2_train = G2[:int(D_mix.shape[0]*0.8),:]
y_train = np.concatenate((G1_train,G2_train),axis=1)

X_test = mel_real_imag_expand(D_mix[int(D_mix.shape[0]*0.8):,:])
G1_test = G1[int(D_mix.shape[0]*0.8):,:]
G2_test = G2[int(D_mix.shape[0]*0.8):,:]
y_test = np.concatenate((G1_test,G2_test),axis=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=87)

# normalize the data
G_max = np.max(y_train)
X_train = min_max_norm(X_train)
y_train = y_train / G_max
X_val = min_max_norm(X_val)
y_val = y_val / G_max
X_test = min_max_norm(X_test)
y_test = y_test / G_max

########################################
# gain method separation model
## import keras modules
from keras.layers import BatchNormalization,Dropout,Dense,Input,LeakyReLU
from keras import backend as K
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.models import Model
from keras.utils import plot_model
from keras.initializers import he_normal
from keras.models import model_from_json
from keras import optimizers

if TRAIN:
    # structure
    n_input_dim = X_train.shape[1]
    n_output_dim = y_train.shape[1]

    n_hidden1 = 2049
    n_hidden2 = 500
    n_hidden3 = 180

    InputLayer1 = Input(shape=(n_input_dim,), name="InputLayer")
    InputLayer2 = BatchNormalization(momentum=0.6)(InputLayer1)

    HiddenLayer1_1 = Dense(n_hidden1, name="H1", activation='relu', kernel_initializer=he_normal(seed=27))(InputLayer2)
    HiddenLayer1_2 = BatchNormalization(momentum=0.6)(HiddenLayer1_1)

    HiddenLayer2_1 = Dense(n_hidden2, name="H2", activation='relu', kernel_initializer=he_normal(seed=42))(HiddenLayer1_2)
    HiddenLayer2_2 = BatchNormalization(momentum=0.6)(HiddenLayer2_1)

    HiddenLayer3_1 = Dense(n_hidden3, name="H3", activation='relu', kernel_initializer=he_normal(seed=65))(HiddenLayer2_2)
    HiddenLayer3_2 = BatchNormalization(momentum=0.6)(HiddenLayer3_1)

    HiddenLayer2__1 = Dense(n_hidden2, name="H2_R", activation='relu', kernel_initializer=he_normal(seed=42))(HiddenLayer3_2)
    HiddenLayer2__2 = BatchNormalization(momentum=0.6)(HiddenLayer2__1)

    HiddenLayer1__1 = Dense(n_hidden1, name="H1_R", activation='relu', kernel_initializer=he_normal(seed=27))(HiddenLayer2__2)
    HiddenLayer1__2 = BatchNormalization(momentum=0.6)(HiddenLayer1__1)

    OutputLayer = Dense(n_output_dim, name="OutputLayer", kernel_initializer=he_normal(seed=62))(HiddenLayer1__2)

    model = Model(inputs=[InputLayer1], outputs=[OutputLayer])
    opt = optimizers.Adam(lr=3*1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0001, amsgrad=False)
    model.compile(loss='mse', optimizer=opt)

    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    model.summary()

    tensorboard = TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True)

    # train the model
    hist = model.fit(X_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=([X_val], [y_val]),
                     callbacks=[tensorboard])

    results = model.evaluate(X_test, y_test, batch_size=len(y_test))
    print('Test loss:%3f' % results)

    # serialize model to JSON

    os.system('mkdir save')
    model_json = model.to_json()
    with open("save/model1.json", 'w') as f:
        f.write(model_json)
    # serialize weights to HDF5
    model.save_weights("save/model1.h5")
    print("Saved model to disk")

if TEST:
    # load json and create model
    with open('save/model1.json','r') as f:
        loaded_model_json = f.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("save/model1.h5")
    print("Loaded model from disk")

    path = '../../data/audio/audio/mix_audio.wav'

    with open(path, 'rb') as f:
        test_data, test_sr = librosa.load(path, sr=None)

    F = window_fft(test_data, fft_size, step_size)
    M = mel_real_imag_expand(time_to_mel(test_data,test_sr,fft_size,n_mel,step_size))

    G = model.predict(min_max_norm(M))
    print(G.shape)
    G_size = G.shape[1]
    G1 = G[:,:G_size//2]
    G2 = G[:,G_size//2:]

    print(G.shape,G1.shape,G2.shape)
    print(G)


    # transfer gain to frequency domain
    F_G1 = rev_gain(G1, G_max, test_sr, fft_size, n_mel)
    F_G2 = rev_gain(G2, G_max, test_sr, fft_size, n_mel)

    F1 = F_G1 * F
    F2 = F_G2 * F

    T1 = rev_window_fft(F1,fft_size,step_size)
    T2 = rev_window_fft(F2,fft_size,step_size)

    T1int = T1 / np.max(T1) * 30000
    wavfile.write("part1.wav", mix_sr, T1int.astype('int16'))

    T2int = T2 / np.max(T2) * 30000
    wavfile.write("part2.wav", mix_sr, T2int.astype('int16'))


if INVERSE_CHECK:
    path = '../../data/audio/audio_train/mix_audio.wav'
    with open(path, 'rb') as f:
        test_data, test_sr = librosa.load(path, sr=None)
    F = window_fft(test_data, fft_size, step_size)
    T = rev_window_fft(F,fft_size, step_size)
    Tint = T / np.max(T) * 30000
    wavfile.write("test1.wav", mix_sr, Tint.astype('int16'))

    M = time_to_mel(test_data,test_sr,fft_size,n_mel,step_size,fmax=8000)
    r_T = mel_to_time(M,test_sr,fft_size,n_mel,step_size)
    Tint = r_T / np.max(r_T) * 30000
    wavfile.write("test2.wav", mix_sr, Tint.astype('int16'))



    G1 = rev_gain(G1,1,test_sr,fft_size,n_mel)
    F = np.real(F) * np.real(G1) + np.imag(F) * np.imag(G1)
    T = rev_window_fft(F,fft_size,step_size)
    Tint = T / np.max(T) * 30000
    wavfile.write("test3.wav", mix_sr, Tint.astype('int16'))
















