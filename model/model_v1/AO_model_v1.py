import sys
sys.path.append('../lib')
import model_AO as AO
from model_ops import ModelMGPU,latest_file
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.models import Model, load_model
from MyGenerator import AudioGenerator
from keras.callbacks import TensorBoard
from keras import optimizers
import os
from model_loss import audio_discriminate_loss as audio_loss


# create AO model
#############################################################
RESTORE = False
# If set true, continue training from last checkpoint
# needed change 1:h5 file name, 2:epochs num, 3:initial_epoch

# super parameters
people_num = 2
epochs = 100
initial_epoch = 0
batch_size = 8 # 4 to feed one 16G GPU
gamma_loss = 0.1
beta_loss = 0.5

# physical devices option to accelerate training process
workers = 1 # num of core
use_multiprocessing = False
NUM_GPU = 1

# PATH
path = './saved_models_AO' # model path
database_dir_path = '../../data/audio/audio_database'
#############################################################

# create folder to save models
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)
    print('create folder to save models')
filepath = path + "/AOmodel-" + str(people_num) + "p-{epoch:03d}-{val_loss:.5f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


#############################################################
# automatically change lr
def scheduler(epoch):
    ini_lr = 0.00005
    lr = ini_lr
    if epoch >= 5:
        lr = ini_lr / 5
    if epoch >= 10:
        lr = ini_lr / 10
    return lr

rlr = LearningRateScheduler(scheduler, verbose=1)
#############################################################
# read train and val file name
# format: mix.npy single.npy single.npy
trainfile = []
valfile = []
with open((database_dir_path+'/dataset_train.txt'), 'r') as t:
    trainfile = t.readlines()
with open((database_dir_path+'/dataset_val.txt'), 'r') as v:
    valfile = v.readlines()
# ///////////////////////////////////////////////////////// #

# the training steps
if RESTORE:
    latest_file = latest_file(path+'/')
    AO_model = load_model(latest_file)
    info = latest_file.strip().split('-')
    initial_epoch = int(info[-2])
else:
    AO_model = AO.AO_model(people_num)

train_generator = AudioGenerator(trainfile,database_dir_path= database_dir_path, batch_size=batch_size, shuffle=True)
val_generator = AudioGenerator(valfile,database_dir_path=database_dir_path, batch_size=batch_size, shuffle=True)

if NUM_GPU > 1:
    parallel_model = ModelMGPU(AO_model,NUM_GPU)
    adam = optimizers.Adam()
    loss = audio_loss(gamma=gamma_loss,beta=beta_loss,num_speaker=people_num)
    parallel_model.compile(loss=loss,optimizer=adam)
    print(AO_model.summary())
    parallel_model.fit_generator(generator=train_generator,
                           validation_data=val_generator,
                           epochs=epochs,
                           workers = workers,
                           use_multiprocessing= use_multiprocessing,
                           callbacks=[TensorBoard(log_dir='./log_AO'), checkpoint, rlr],
                           initial_epoch=initial_epoch
                           )
if NUM_GPU <= 1:
    adam = optimizers.Adam()
    loss = audio_loss(gamma=gamma_loss,beta=beta_loss, num_speaker=people_num)
    AO_model.compile(optimizer=adam, loss=loss)
    print(AO_model.summary())
    AO_model.fit_generator(generator=train_generator,
                           validation_data=val_generator,
                           epochs=epochs,
                           workers = workers,
                           use_multiprocessing= use_multiprocessing,
                           callbacks=[TensorBoard(log_dir='./log_AO'), checkpoint, rlr],
                           initial_epoch=initial_epoch
                           )


























