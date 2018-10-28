import sys
sys.path.append('../lib')
import os
import utils as avp
import model_AO
import librosa
# import for model
from keras import optimizers
#from keras.layers import Dense, Convolution3D, MaxPooling3D, ZeroPadding3D, Dropout, Flatten, BatchNormalization, ReLU
from keras.models import Sequential, model_from_json
from keras import optimizers
from keras.layers import Input, Dense, Convolution2D, Deconvolution2D, Bidirectional
from keras.layers import Dropout, Flatten, BatchNormalization, ReLU, Reshape, Permute
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import LSTM
from keras.initializers import he_normal
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

# options
SHAPE_CHECK = 0
MODEL_CHECK1 = 0
MODEL_CHECK2 = 1


data1,sr1 = librosa.load('../../data/audio/audio_train/trim_audio_train0.wav',sr=16000)
data2,sr2 = librosa.load('../../data/audio/audio_train/trim_audio_train1.wav',sr=16000)
mix = data1*0.5+data2*0.5

if SHAPE_CHECK:
    # check shape
    print(data1.shape)
    D1 = avp.fast_stft(data1)
    print(D1.shape)
    T1 = avp.fast_istft(D1)
    print(T1.shape)

# check model
if MODEL_CHECK1:
    F1 = avp.fast_stft(data1)
    F1 = np.expand_dims(F1, axis=0)
    F2 = avp.fast_stft(data2)
    F2 = np.expand_dims(F2, axis=0)
    FM = avp.fast_stft(mix)
    FM = np.expand_dims(FM, axis=0)

    cRM1 = np.abs(F1)/np.abs(FM)
    cRM1[~np.isfinite(cRM1)] = 0
    cRM2 = np.abs(F2)/np.abs(FM)
    cRM2[~np.isfinite(cRM2)] = 0
    cRM = np.zeros((1,298,257,2,2))
    cRM[0,:,:,:,0] = cRM1
    cRM[0,:,:,:,1] = cRM2
    print('shape of cRM should be (1,298,257,2,2):', cRM.shape)

    input_dim = (298,257,2)
    output_dim = (298,257,2,2)
    filepath = "model-{epoch:02d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model = model_AO.AO_model(2)
    model.fit(FM, cRM,
                 epochs=5,
                 batch_size=2,
                 validation_data=(FM,cRM),
                 shuffle=False,
                 callbacks=[TensorBoard(log_dir='./log'), checkpoint])

    pred_cRM = model.predict(FM)

if MODEL_CHECK2:
    # check data generative function
    people_num = 2
    sample_range = (0,20)

    repo_path = os.path.expanduser('../../data/audio/audio_train')
    X_data,y_data = avp.generate_dataset(sample_range,repo_path,num_speaker=2,verbose=1)
    print('shape of the X data: ',X_data.shape)
    print('shape of the y data: ',y_data.shape)

    # split data to training and validation set
    X_train, X_val, y_train, y_val = train_test_split(X_data,y_data,test_size=0.05,random_state=30)


    # check feeding tensor to model
    input_dim = (298,257,2)
    output_dim = (298,257,2,2)



    # train and load model
    RESTORE = False
    path = './saved_models_AO'
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print('create folder to save models')

    filepath = path + "/AOmodel-" + str(people_num) + "p-{epoch:03d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    if RESTORE:
        AO_model = load_model('./saved_models_AO/***.h5')
    else:
        AO_model = model_AO.AO_model(people_num)
        AO_model.fit(X_train, y_train,
                     epochs=200,
                     batch_size=4,
                     validation_data=(X_val, y_val),
                     shuffle=False,
                     callbacks=[TensorBoard(log_dir='./log_AO'), checkpoint],
                     initial_epoch=0)

    #pred_cRM = model.predict(X_data)







