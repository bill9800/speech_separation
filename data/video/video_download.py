from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import datetime
sys.path.append("../lib")
import AVHandler as avh
import pandas as pd



def video_download(loc,cat,start_idx,end_idx):
    # Only download the video from the link
    # loc        | the location for downloaded file
    # v_name     | the name for the video file
    # cat        | the catalog with audio link and time
    # start_idx  | the starting index of the video to download
    # end_idx    | the ending index of the video to download

    for i in range(start_idx,end_idx):
        command = 'cd %s;' % loc
        f_name = str(i)
        link = avh.m_link(cat.loc[i, 'link'])
        start_time = cat.loc[i, 'start_time']
        end_time = start_time + 3.0
        start_time = datetime.timedelta(seconds=start_time)
        end_time = datetime.timedelta(seconds=end_time)
        command += 'ffmpeg -i $(youtube-dl -f ”mp4“ --get-url ' + link + ') ' + '-c:v h264 -c:a copy -ss %s -to %s %s.mp4' \
                % (start_time, end_time, f_name)
        #command += 'ffmpeg -i %s.mp4 -r 25 %s.mp4;' % (f_name,'clip_' + f_name) #convert fps to 25
        #command += 'rm %s.mp4' % f_name
        os.system(command)

def generate_frames(loc,start_idx,end_idx):
    # get frames for each video clip
    # loc        | the location of video clip
    # v_name     | v_name = 'clip_video_train'
    # start_idx  | the starting index of the training sample
    # end_idx    | the ending index of the training sample

    avh.mkdir('frames')
    for i in range(start_idx, end_idx):
        command = 'cd %s;' % loc
        f_name = str(i)
        command += 'ffmpeg -i %s.mp4 -y -f image2  -vframes 75 ../frames/%s-%%02d.jpg' % (f_name, f_name)
        os.system(command)


def download_video_frames(loc,cat,start_idx,end_idx,rm_video):
    # Download each video and convert to frames immediately, can choose to remove video file
    # loc        | the location for downloaded file
    # cat        | the catalog with audio link and time
    # start_idx  | the starting index of the video to download
    # end_idx    | the ending index of the video to download
    # rm_video   | boolean value for delete video and only keep the frames

    avh.mkdir('frames')
    for i in range(start_idx, end_idx + 1):
        command = 'cd %s;' % loc
        f_name = str(i)
        link = avh.m_link(cat.loc[i, 'link'])
        start_time = cat.loc[i, 'start_time']
        end_time = start_time + 3.0
        start_time = datetime.timedelta(seconds=start_time)
        end_time = datetime.timedelta(seconds=end_time)
        command += 'ffmpeg -i $(youtube-dl -f ”mp4“ --get-url ' + link + ') ' + '-c:v h264 -c:a copy -ss %s -to %s %s.mp4;' \
                   % (start_time, end_time, f_name)
        #ommand += 'ffmpeg -i %s.mp4 -r 25 %s.mp4;' % (f_name, 'clip_' + f_name)  # convert fps to 25
        #command += 'rm %s.mp4;' % f_name

        #converts to frames
        #command += 'ffmpeg -i %s.mp4 -y -f image2  -vframes 75 ../frames/%s-%%02d.jpg;' % (f_name, f_name)
        command += 'ffmpeg -i %s.mp4 -vf fps=25 ../frames/%s-%%02d.jpg;' % (f_name, f_name)
        #command += 'ffmpeg -i %s.mp4 ../frames/%sfr_%%02d.jpg;' % ('clip_' + f_name, f_name)

        if rm_video:
            command += 'rm %s.mp4' % f_name
        os.system(command)

avh.mkdir('video_train')
cat_train = pd.read_csv('../audio/catalog/avspeech_train.csv')

# download video , convert to images separately
#avh.video_download(loc='video_train',v_name='video_train',cat=cat_train,start_idx=2,end_idx=4)
#avh.generate_frames(loc='video_train',v_name='clip_video_train',start_idx=2,end_idx=4)

# download each video and convert to frames immediately
download_video_frames(loc='video_train',cat=cat_train,start_idx=1405,end_idx=2000,rm_video=True)
