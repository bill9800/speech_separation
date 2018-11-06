import sys
sys.path.append("../../model/lib")
import os
import librosa
import numpy as np
import utils
import operator
import itertools
import time

# Parameter
SAMPLE_RANGE = (0,350) # data usage to generate database
TEST_RANGE = (350,550) # data usage to generate database
REPO_PATH = os.path.expanduser("./audio_train")
TRAIN = 1
TEST = 0

# time measure decorator
def timit(func):
    def cal_time(*args,**kwargs):
        tic = time.time()
        result = func(*args,**kwargs)
        tac = time.time()
        print(func.__name__,'running time: ',(tac-tic),'ms')
        return result
    return cal_time

# create directory to store database
def init_dir():
    if not os.path.isdir('./audio_database'):
        os.mkdir('./audio_database')

    if not os.path.isdir('./audio_database/mix'):
        os.mkdir('./audio_database/mix')

    if not os.path.isdir('./audio_database/single'):
        os.mkdir('./audio_database/single')

    if not os.path.isdir('./audio_database/crm'):
        os.mkdir('./audio_database/crm')


def generate_path_list(sample_range=SAMPLE_RANGE,repo_path=REPO_PATH):
    '''

    :param sample_range:
    :param repo_path:
    :return: 2D array with idx and path
    '''
    audio_path_list = []

    for i in range(sample_range[0],sample_range[1]):
        path = repo_path + '/trim_audio_train%d.wav'%i
        if os.path.exists(path):
            audio_path_list.append((i,path))
    print('length of the path list: ',len(audio_path_list))
    return audio_path_list

# data generate function
def single_audio_to_npy(audio_path_list,fix_sr=16000):
    for idx,path in audio_path_list:
        data, _ = librosa.load(path, sr=fix_sr)
        data = utils.fast_stft(data)
        name = 'single-%05d'%idx
        np.save(('audio_database/single/%s.npy'%name),data)

def generate_mix_sample(audio_path_list,num_speaker,fix_sr=16000,verbose=0):
    '''
    generate mix sample from audios in the list

    :param audio_path_list: list contains path of the wav audio file
    :param num_speaker: specify the task for speech separation
    :param fix_sr: fix sample rate
    '''
    # initiate variables
    # shape of F_mix = (298,257,2)
    # shpae of crm = (298,257,2)
    data_list = []
    F_list = []  # STFT list for each sample
    cRM_list = []

    mix_name = "mix"
    crm_name = "crm"
    post_name = ""

    # import data
    for i in range(num_speaker):
        idx,path =audio_path_list[i]
        post_name += "-%05d"%idx
        data, _ = librosa.load(path,sr=fix_sr)
        data_list.append(data)

    # create mix audio according to mix rate
    mix_rate = 1.0 / float(num_speaker)
    mix = np.zeros(shape=data_list[0].shape)
    for data in data_list:
        mix += data*mix_rate
    # transfrom data via STFT and several preprocessing function
    for i in range(num_speaker):
        F = utils.fast_stft(data_list[i],power=False)
        F_list.append(F)
    F_mix = utils.fast_stft(mix,power=False)
    # create cRM for each speaker and fill into y_sample
    for i in range(num_speaker):
        cRM_list.append(utils.fast_cRM(F_list[i],F_mix))

    # return values
    if verbose == 1:
        print('shape of X: ',F_mix.shape)
        for i in range(len(cRM_list)):
            print('shape of cRM%s :'%i,cRM_list[i].shape)

    # save record in txt
    mix_name += post_name
    crm_name += post_name

    # write txt
    with open('audio_database/dataset.txt','a') as f:
        f.write(mix_name+".npy")
        for i in range(len(cRM_list)):
            line = " " + crm_name + ("-%05d"%audio_path_list[i][0]) + ".npy"
            f.write(line)
        f.write("\n")

    # save file as npy
    np.save(('audio_database/mix/%s.npy'%mix_name), F_mix)
    for i in range(len(cRM_list)):
        name = crm_name + ("-%05d"%audio_path_list[i][0])
        np.save(('audio_database/crm/%s.npy'%name), cRM_list[i])

@timit
def generate_dataset(sample_range,repo_path,num_speaker=2):
    '''
    A function to generate dataset
    :param sample_range: range of the sample to create the dataset
    :param repo_path: audio repository
    :param num_speaker: number of speaker to separate
    :return: X_data, y_data
    '''
    audio_path_list = generate_path_list(sample_range,repo_path)
    num_data = 0

    combinations = itertools.combinations(audio_path_list,num_speaker)
    for combo in combinations:
        num_data += 1
        generate_mix_sample(combo,num_speaker)

    print('number of the data generated: ',num_data)

@timit
def train_test_split(num_start = 0,num_data = 50000,database_txt_path="audio_database/dataset.txt",
                     train_txt_path="audio_database/dataset_train.txt",test_txt_path="audio_database/dataset_val.txt",test_rate=0.01):
    step = 1 // test_rate
    count = 0
    with open(database_txt_path,'r') as f:
        for i in range(num_start):
            _ = f.read()
        train_txt = open(train_txt_path,'a')
        test_txt = open(test_txt_path,'a')
        for line in f:
            if count > num_data :
                break
            count+=1
            if count%step==0:
                test_txt.write(line)
                continue
            train_txt.write(line)
        train_txt.close()
        test_txt.close()

@timit
def create_testset(num_start = 53400,num_data=12375,database_txt_path="audio_database/dataset.txt",test_txt_path="audio_database/dataset_test.txt"):
    count = 0
    with open(database_txt_path,'r') as f:
        test_txt = open(test_txt_path, 'a')
        for i in range(num_start):
            f.readline()
        for line in f:
            if count > num_data:
                break
            count += 1
            test_txt.write(line)
        test_txt.close()

#####################################
# create directory
init_dir()

## generate training dataset
if TRAIN:
    # generate path
    audio_path_list = generate_path_list()
    # generate single data
    single_audio_to_npy = single_audio_to_npy(audio_path_list)
    # generate mix data
    generate_dataset(sample_range=SAMPLE_RANGE,repo_path=REPO_PATH)
    # split train and test txt instructor file
    train_test_split()

## generate testing dataset
if TEST:
    audio_path_list = generate_path_list(sample_range=TEST_RANGE,repo_path=REPO_PATH)
    single_audio_to_npy(audio_path_list)
    generate_dataset(sample_range=TEST_RANGE, repo_path=REPO_PATH)
    create_testset()






















