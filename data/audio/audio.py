# Before running, make sure avspeech_train.csv and avspeech_test.csv are in catalog.
# if not, see the requirement.txt
# download and preprocess the data from AVspeech dataset
import sys
sys.path.append("../lib")
import AVHandler as avh
import pandas as pd

def m_link(youtube_id):
    # return the youtube actual link
    link = 'https://www.youtube.com/watch?v='+youtube_id
    return link

def m_audio(loc,name,cat,start_idx,end_idx):
    # make concatenated audio following by the catalog from AVSpeech
    # loc       | the location for file to store
    # name      | name for the wav mix file
    # cat       | the catalog with audio link and time
    # start_idx | the starting index of the audio to download and concatenate
    # end_idx   | the ending index of the audio to download and concatenate

    for i in range(start_idx,end_idx):
        f_name = name+str(i)
        link = m_link(cat.loc[i,'link'])
        start_time = cat.loc[i,'start_time']
        end_time = start_time + 3.0
        avh.download(loc,f_name,link)
        avh.cut(loc,f_name,start_time,end_time)

cat_train = pd.read_csv('catalog/avspeech_train.csv')
#cat_test = pd.read_csv('catalog/avspeech_test.csv')

# create 80000-90000 audios data from 290K
avh.mkdir('audio_train')
m_audio('audio_train','audio_train',cat_train,80000,80500)

