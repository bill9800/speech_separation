## evaluate the model and generate the prediction
import sys
sys.path.append('../lib')
from keras.models import load_model
from MyGenerator import AudioGenerator
from model_ops import ModelMGPU
import os
import scipy.io.wavfile as wavfile
import numpy as np
import utils
# super parameters
people_num = 2
NUM_GPU = 1

# PATH
model_path = './saved_models_AO_with_norm/AOmodel-2p-015-0.02258.h5'
dir_path = './pred/'
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)
database_path = '../../data/audio/audio_database/mix/'


# load data
testfiles = []
with open('../../data/audio/audio_database/dataset_train.txt', 'r') as f:
    testfiles = f.readlines()
    # get num_data
    testfiles = [testfiles[1],testfiles[2]]
def parse_X_data(testfile,num_people=people_num,database_path=database_path):
    line = testfile.split() # get each name of file for one testset
    mix_str = line[0]
    name_list = mix_str.replace('.npy','')
    names = name_list.split('-')
    single_idxs = []
    for i in range(num_people):
        single_idxs.append(names[i+1])
    file_path = database_path + mix_str
    mix = np.load(file_path)

    return mix,single_idxs


# predict data
AO_model = load_model(model_path)
if NUM_GPU > 1:
    parallel_model = ModelMGPU(AO_model,NUM_GPU)
    for line in testfiles:
        mix,single_idxs = parse_X_data(line)
        mix_expand = np.expand_dims(mix, axis=0)
        cRMs = parallel_model.predict(mix_expand)
        cRMs = cRMs[0]
        for i in range(len(cRMs)):
            cRM = cRMs[:,:,:,i]
            assert cRM.shape == (298,257,2)
            F = mix * cRM
            T = utils.fast_istft(F)
            Tint = T/np.max(T) * 30000
            filename = dir_path+single_idxs[i]+'.wav'
            wavfile.write(filename,16000,Tint.astype('int16'))


if NUM_GPU <= 1:
    for line in testfiles:
        mix,single_idxs = parse_X_data(line)
        print('mix:',mix)
        mix_expand = np.expand_dims(mix,axis=0)
        print("mix_expand:",mix_expand)
        cRMs = AO_model.predict(mix_expand)
        cRMs = cRMs[0]
        for i in range(people_num):
            cRM = cRMs[:,:,:,i]
            assert cRM.shape == (298,257,2)
            print("cRM50:",cRM[50])
            print("cRM100:",cRM[100])
            # invert the compression cRM from sigmoid function
            cRM = np.log(cRM/(1-cRM))
            print("cRM_reverse50:",cRM[50])
            print("cRM_reverse100:", cRM[100])
            F = mix * cRM
            print("F50:",F[50])
            print("F100:",F[100])
            T = utils.fast_istft(F)
            print("T50:",T[50])
            print("T100:",T[100])
            Tint = T/np.max(T) * 30000
            filename = dir_path+single_idxs[i]+'.wav'
            wavfile.write(filename,16000,Tint.astype('int16'))


















