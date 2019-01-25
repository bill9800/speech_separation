import sys
sys.path.append("../../model/lib")
import os
import librosa
import numpy as np
import utils
import itertools
import time
import random
import math
import scipy.io.wavfile as wavfile

# Parameter
SAMPLE_RANGE = (0,20) # data usage to generate database
WAV_REPO_PATH = os.path.expanduser("./norm_audio_train")
DATABASE_REPO_PATH = 'AV_model_database'
FRAME_LOG_PATH = '../video/valid_frame.txt'
NUM_SPEAKER = 2
MAX_NUM_SAMPLE = 50000

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
def init_dir(path = DATABASE_REPO_PATH ):
    if not os.path.isdir(path):
        os.mkdir(path)

    if not os.path.isdir('%s/mix'%path):
        os.mkdir('%s/mix'%path)

    if not os.path.isdir('%s/single'%path):
        os.mkdir('%s/single'%path)

    if not os.path.isdir('%s/crm'%path):
        os.mkdir('%s/crm'%path)

    if not os.path.isdir('%s/mix_wav'%path):
        os.mkdir('%s/mix_wav'%path)

@timit
def generate_path_list(sample_range=SAMPLE_RANGE,repo_path=WAV_REPO_PATH,frame_path=FRAME_LOG_PATH):
    '''

    :param sample_range:
    :param repo_path:
    :return: 2D array with idx and path (idx_wav,path_wav)
    '''
    audio_path_list = []
    frame_set = set()

    with open(frame_path,'r') as f:
        frames = f.readlines()
    
    for i in range(len(frames)):
        frame = frames[i].replace('\n','').replace('frame_','')
        frame_set.add(int(frame))

    for i in range(sample_range[0],sample_range[1]):
        print('\rchecking...%d'%int(frame),end='')
        path = repo_path + '/trim_audio_train%d.wav'%i
        if os.path.exists(path) and (i in frame_set):
            audio_path_list.append((i,path))
    print('\nlength of the path list: ',len(audio_path_list))
    return audio_path_list

# data generate function
def single_audio_to_npy(audio_path_list,database_repo=DATABASE_REPO_PATH,fix_sr=16000):
    for idx,path in audio_path_list:
        print('\rsingle npy generating... %d'%((idx/len(audio_path_list))*100),end='')
        data, _ = librosa.load(path, sr=fix_sr)
        data = utils.fast_stft(data)
        name = 'single-%05d'%idx
        with open('%s/single_TF.txt'%database_repo,'a') as f:
            f.write('%s.npy'%name)
            f.write('\n')
        np.save(('%s/single/%s.npy'%(database_repo,name)),data)
    print()


# split single TF data to different part in order to mix
def split_to_mix(audio_path_list,database_repo=DATABASE_REPO_PATH,partition=2):
    # return split_list : (part1,part2,...)
    # each part : (idx,path)
    length = len(audio_path_list)
    part_len = length // partition
    head = 0
    part_idx = 0
    split_list = []
    while((head+part_len)<length):
        part = audio_path_list[head:(head+part_len)]
        split_list.append(part)
        with open('%s/single_TF_part%d.txt'%(database_repo,part_idx),'a') as f:
            for idx, _ in part:
                name = 'single-%05d' % idx
                f.write('%s.npy' % name)
                f.write('\n')
        head += part_len
        part_idx += 1
    return split_list

# mix single TF data
def all_mix(split_list,database_repo=DATABASE_REPO_PATH,partition=2):
    assert len(split_list) == partition
    print('mixing data...')
    num_mix = 1
    num_mix_check = 0
    for part in split_list:
        num_mix *= len(part)
    print ('number of mix data; ',num_mix)

    part_len = len(split_list[-1])
    idx_list = [x for x in range(part_len)]
    combo_idx_list = itertools.product(idx_list,repeat=partition)
    for combo_idx in combo_idx_list:
        num_mix_check +=1
        single_mix(combo_idx,split_list,database_repo)
        print('\rnum of completed mixing audio : %d'%num_mix_check,end='')  
    print()
# mix several wav file and store TF domain data with npy
def single_mix(combo_idx,split_list,database_repo):
    assert len(combo_idx) == len(split_list)
    mix_rate = 1.0 / float(len(split_list))
    wav_list = []
    prefix = "mix"
    mid_name = ""

    for part_idx in range(len(split_list)):
        idx,path = split_list[part_idx][combo_idx[part_idx]]
        wav, _ = librosa.load(path, sr=16000)
        wav_list.append(wav)
        mid_name += '-%05d' % idx

    # mix wav file
    mix_wav = np.zeros_like(wav_list[0])
    for wav in wav_list:
        mix_wav += wav * mix_rate

    # save mix wav file
    wav_name = prefix+mid_name+'.wav'
    wavfile.write('%s/mix_wav/%s'%(database_repo,wav_name),16000,mix_wav)

    # transfer mix wav to TF domain
    F_mix = utils.fast_stft(mix_wav)
    name = prefix+mid_name+".npy"
    store_path = '%s/mix/%s'%(database_repo,name)

    # save mix as npy file
    np.save(store_path,F_mix)

    # save mix log
    with open('%s/mix_log.txt'%database_repo,'a') as f:
        f.write(name)
        f.write("\n")


def all_crm(mix_log_path,database_repo=DATABASE_REPO_PATH):
    with open(mix_log_path,'r') as f:
        mix_list = f.read().splitlines()

    for mix in mix_list:
        mix_path = '%s/mix/%s' % (database_repo, mix)
        mix = mix.replace(".npy","")
        mix = mix.replace("mix-","")
        idx_str_list = mix.split("-")
        single_crm(idx_str_list,mix_path,database_repo)

def single_crm(idx_str_list,mix_path,database_repo):
    F_mix = np.load(mix_path)
    mid_name = ""
    mix_name = "mix"
    dataset_line = ""

    for idx in idx_str_list:
        mid_name += "-%s"%idx
        mix_name += "-%s"%idx
    mix_name += '.npy'
    dataset_line += mix_name

    for idx in idx_str_list:
        single_name = 'single-%s.npy'%idx
        path = '%s/single/%s'%(database_repo,single_name)
        F_single = np.load(path)
        cRM = utils.fast_cRM(F_single,F_mix)

        last_name = '-%s'%idx
        cRM_name = 'crm' + mid_name + last_name + '.npy'

        # save crm to npy
        store_path = '%s/crm/%s'%(database_repo,cRM_name)
        np.save(store_path,cRM)

        # save crm information to log
        with open('%s/crm_log.txt'%database_repo, 'a') as f:
            f.write(cRM_name)
            f.write('\n')
        dataset_line += (" "+cRM_name)

    # write in database log
    with open('%s/dataset.txt'%database_repo,'a') as f:
        f.write(dataset_line)
        f.write('\n')


def train_test_split(dataset_log_path,data_range=[0,50000],test_ratio=0.1,shuffle=True,database_repo=DATABASE_REPO_PATH):
    with open(dataset_log_path,'r') as f:
        data_log = f.read().splitlines()

    if data_range[1]> len(data_log):
        data_range[1] = len(data_log)-1
    samples = data_log[data_range[0]:data_range[1]]
    if shuffle:
        random.shuffle(samples)

    length = len(samples)
    mid = int(math.floor(test_ratio*length))
    test = samples[:mid]
    train = samples[mid:]

    with open('%s/dataset_train.txt'%database_repo,'a') as f:
        for line in train:
            f.write(line)
            f.write('\n')

    with open('%s/dataset_val.txt' % database_repo, 'a') as f:
        for line in test:
            f.write(line)
            f.write('\n')

if __name__ == "__main__":
    init_dir()
    audio_path_list = generate_path_list()
    single_audio_to_npy(audio_path_list)
    split_list = split_to_mix(audio_path_list,partition=NUM_SPEAKER)
    all_mix(split_list,partition=NUM_SPEAKER)

    mix_log_path = '%s/mix_log.txt'%DATABASE_REPO_PATH
    all_crm(mix_log_path)

    dataset_log_path ='%s/dataset.txt'%DATABASE_REPO_PATH
    train_test_split(dataset_log_path,data_range=[0,MAX_NUM_SAMPLE])













































































































