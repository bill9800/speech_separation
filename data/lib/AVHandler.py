import os
import librosa
import scipy.io.wavfile as wavfile
import numpy as np
# A file for downloading files and handling audio and video 

# command line functions #


def mkdir(dir_name,loc=''):
    # make directory use command line
    # dir_name  | name of the directory
    # loc       | the location for the directory to be created
    command = ""
    if loc != '':
        command += "cd %s" % loc
    command += 'mkdir ' + dir_name
    os.system(command)

def m_link(youtube_id):
    # return the youtube actual link
    link = 'https://www.youtube.com/watch?v='+youtube_id
    return link

def download(loc,name,link,sr=16000,type='audio'):
    # download audio from the link
    # loc   | the location for downloaded file
    # name  | the name for the audio file
    # link  | the link to downloaded by youtube-dl
    # type  | the type of downloaded file


    if type == 'audio':
        # download wav file from the youtube link
        command = 'cd %s;' % loc
        command += 'youtube-dl -x --audio-format wav -o o' + name + '.wav ' + link + ';'
        command += 'ffmpeg -i o%s.wav -ar %d -ac 1 %s.wav;' % (name,sr,name)
        command += 'rm o%s.wav' % name
        os.system(command)



def cut(loc,name,start_time,end_time):
    # trim the audio/video by sox
    # loc         | the location of the file
    # name        | the name of file to trim
    # start_time  | the start time of the audio segment
    # end_time    | the end time of the audio segment
    length = end_time - start_time
    command = 'cd %s;' % loc
    command += 'sox %s.wav trim_%s.wav trim %s %s;' % (name,name,start_time,length)
    command += 'rm %s.wav' % name
    os.system(command)


def conc(loc,name,trim_clean=False):
    # concatenate the data in the loc (trim*.wav)
    command = 'cd %s;' % loc
    command += 'sox --combine concatenate trim_*.wav -o %s.wav;' % name
    if trim_clean:
    	command += 'rm trim_*.wav;'
    os.system(command)


def mix(loc,name,file1,file2,start,end,trim_clean=False):
    # mix the audio/video via sox
    # loc         | location of the mix files
    # name        | output name of wav
    # file1       | first file to mix
    # file2       | second file to mix
    # start       | mixture starting time
    # end         | mixture end time
    # trim_clean  | delete the trim file or not
    command = 'cd %s;' % loc
    cut(loc,file1,start,end)
    cut(loc,file2,start,end)
    trim1 = '%s/trim_%s.wav' % (loc,file1)
    trim2 = '%s/trim_%s.wav' % (loc,file2)
    with open(trim1, 'rb') as f:
        wav1, wav1_sr = librosa.load(trim1, sr=None)  # time series data,sample rate
    with open(trim2, 'rb') as f:
        wav2, wav2_sr = librosa.load(trim2, sr=None)

    # compress the audio to same volume level
    wav1 = wav1 / np.max(wav1)
    wav2 = wav2 / np.max(wav2)
    assert wav1_sr == wav2_sr
    mix_wav = wav1*0.5+wav2*0.5

    path = '%s/%s.wav' % (loc,name)
    wavfile.write(path,wav1_sr,mix_wav)
    if trim_clean:
        command += 'rm trim_%s.wav;rm trim_%s.wav;' % (file1,file2)
    os.system(command)









