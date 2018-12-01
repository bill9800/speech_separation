import numpy as np
import librosa
import sys
sys.path.append('../model/lib')
import utils
import os

TRUE_REPO = "/"
PRED_REPO = "/"

files = os.listdir(PRED_REPO)

def mse(true_path,pred_path):
    true_data,_ = librosa.load(true_path)
    true_stft = utils.stft(true_data)
    pred_data,_ = librosa.load(pred_path)
    pred_stft = utils.stft(pred_data)
    return np.mean(np.square(np.subtract(true_stft,pred_stft)))

count = 0
mse_sum = 0

for file in files:
    pred_path = PRED_REPO + file
    speaker = file.replace(".wav","").split("-")[2]
    true_path = TRUE_REPO+"trim_audio_train%d.wav"%int(speaker)
    single_mse = mse(true_path,pred_path)
    mse_sum += single_mse
    with open('mse_result.txt', 'a') as f:
        f.write(file+" "+str(single_mse)+"\n")
    count +=1


print('avg mse:',mse_sum/count)
