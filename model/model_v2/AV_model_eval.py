## evaluate the model and generate the prediction
import sys
sys.path.append('../lib')
from keras.models import load_model
from model_ops import ModelMGPU
import os
import scipy.io.wavfile as wavfile
import numpy as np
import utils
import tensorflow as tf
# super parameters
people_num = 2
NUM_GPU = 1

# PATH
model_path = './saved_models_AO_with_norm/AOmodel-2p-015-0.02258.h5'
dir_path = './pred/'
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)
database_path = '../../data/audio/audio_database/mix/'
face_path = '../../data/video/face_emb/'

# load data
testfiles = []
with open('../../data/AV_log/AVdataset_train.txt', 'r') as f:
    testfiles = f.readlines()

def parse_X_data(line,num_people=people_num,database_path=database_path,face_path=face_path):
    parts = line.split() # get each name of file for one testset
    mix_str = parts[0]
    name_list = mix_str.replace('.npy','')
    name_list = name_list.replace('mix-','',1)
    names = name_list.split('-')
    single_idxs = []
    for i in range(num_people):
        single_idxs.append(names[i])
    file_path = database_path + mix_str
    mix = np.load(file_path)
    face_embs = np.zeros((1,75,1,1792,num_people))
    for i in range(num_people):
        face_embs[1,:,:,:,i] = np.load(face_path+"%05d_face_emb.npy"%single_idxs[i])

    return mix,single_idxs,face_embs


# predict data
AV_model = load_model(model_path,custom_objects={"tf": tf})
if NUM_GPU > 1:
    parallel_model = ModelMGPU(AV_model,NUM_GPU)
    for line in testfiles:
        mix,single_idxs,face_embs = parse_X_data(line)
        mix_expand = np.expand_dims(mix, axis=0)
        cRMs = parallel_model.predict([mix_expand,face_embs])
        cRMs = cRMs[0]
        prefix = ""
        for idx in single_idxs:
            prefix += idx + "-"
        for i in range(len(cRMs)):
            cRM = cRMs[:,:,:,i]
            assert cRM.shape == (298,257,2)
            F = utils.fast_icRM(mix,cRM)
            T = utils.fast_istft(F,power=False)
            filename = dir_path+prefix+str(single_idxs[i])+'.wav'
            wavfile.write(filename,16000,T)


if NUM_GPU <= 1:
    for line in testfiles:
        mix, single_idxs, face_embs = parse_X_data(line)
        mix_expand = np.expand_dims(mix, axis=0)
        cRMs = AV_model.predict([mix_expand, face_embs])
        cRMs = cRMs[0]
        prefix = ""
        for idx in single_idxs:
            prefix += idx + "-"
        for i in range(people_num):
            cRM = cRMs[:,:,:,i]
            assert cRM.shape == (298,257,2)
            F = utils.fast_icRM(mix,cRM)
            T = utils.fast_istft(F,power=False)
            filename = dir_path+prefix+single_idxs[i]+'.wav'
            wavfile.write(filename,16000,T)
















