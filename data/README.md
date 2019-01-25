# Instruction for generating data

Following are the steps to generate training and testing data.  There are several parameters to change in order to match different purpose. 

## Step 1 - Setting log files

1. Get into **data/audio/catalog** and download **train.csv** and **test.csv** from [AVspeech](https://looking-to-listen.github.io/avspeech/download.html)
2. Add **link,start_time,end_time,pos_x,pos_y** to the head of the csv files. (for pandas) 

## Step 2 - Download audio data

1.  Open **data/audio/audio_downloader.py** 

2.  Set the csv file to read, modify **cat_train** or **cat_test**

3.  Set the range of the data in AVspeech dataset. For example from 0-20:

   ```python
   m_audio('audio_train','audio_train',cat_train,0,20)
   ```

4. Run it. Generated audio files will contains in **data/audio_train/**

## Step 3 - Normalize audio data

> This step is not necessary, just to balance the volume before mixing audio data

1. Open **data/audio/audio_norm.py**

2. Change the dataset range, must not larger than downloaded audio range. For example (0,20):

   ```python
   RANGE = (0,20)
   ```

3. Run it. Generated normalized audio files will contains in **data/audio/norm_audio_train**/

## Step 4 - Download visual data 

> This step will download video and extract 75 frames (25FPS) corresponding to the 3 second audio data.

1. Open **data/video/video_download.py**

2. Change **start_idx** and **end_idx** corresponding to the range of the audio data. For example  (0,20):

   ```python
   download_video_frames(loc='video_train',cat=cat_train,start_idx=0,end_idx=20,rm_video=True)
   ```

3. Run it. Generated raw frames will contains in **data/video/frames/**

## Step 5 - Detect and Crop face 

> This step requires tensorflow and mtcnn package, make sure these are installed

1. Open **data/video/MTCNN_detect.py**

2. Change **detect_range** corresponding to the range of data. For example (0,20):

   ```python
   detect_range = (0,20)
   ```

3. Run it. All detected faces will generate in **data/video/face_input/**

4. Open **data/video/frame_inspector.py**

5. Change **inspect_range** to the range of data. For example (0,20):

   ```python
   inspect_range = (0,20)
   ```

6. Run it. This process will delete the series of frames which are not valid, such as wrong detection or missing frames. Valid faces will include in **valid_frame.txt**

## Step 6 - Create audio database 

1. Open **data/audio/build_audio_database_v2.py**
2. Modify **SAMPLE_RANGE** corresponding to range of data, modify **MAX_NUM_SAMPLE** to limit the number of generation data. **NUM_SPEAKER** is related to the number of people we want to separated.
3. Run it. All generated files will contains in **data/audio/AV_model_database/**

## Step 7 - Generate log file for data generator 

1. Open **data/AV_log/gentxtnew.py**
2. Run it. It will generate necessary file for data generator.

## Step 8 - Generate face embedding feature 

1. Get into **mode/pretrain_model/FaceNet_keras**
2. download [keras facenet model](https://github.com/nyoki-mtl/keras-facenet) as facenet_keras.h5 
3. Open **model/pretrain_model/pretrain_load_test.py**
4. Change the parameters for different path. 
5. Run it. Face Embedding numpy files will store in **face1022_emb**

















