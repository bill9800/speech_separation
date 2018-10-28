import sys
sys.path.append('../lib')
import model_AO as AO
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.models import Model, load_model
from MyGenerator import AudioGenerator
from keras.callbacks import TensorBoard
import os

# create AO model
#############################################################
RESTORE = True
# If set true, continue training from last checkpoint
# needed change 1:h5 file name, 2:epochs num, 3:initial_epoch

# super parameters
people_num = 2
epochs = 20
initial_epoch = 0
batch_size = 2
#############################################################

# create folder to save models
path = './saved_models_AO'
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)
    print('create folder to save models')
filepath = path + "/AOmodel-" + str(people_num) + "p-{epoch:03d}-{val_loss:.5f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


#############################################################
# automatically change lr
def scheduler(epoch):
    ini_lr = 0.001
    lr = ini_lr
    if epoch == 5:
        lr = ini_lr / 5
    if epoch == 10:
        lr = ini_lr / 10
    return lr

rlr = LearningRateScheduler(scheduler, verbose=1)
#############################################################
# read train and val file name
# format: mix.npy single.npy single.npy
trainfile = []
valfile = []
with open('../../data/audio/audio_database/dataset_train.txt', 'r') as t:
    trainfile = t.readlines()
with open('../../data/audio/audio_database/dataset_val.txt', 'r') as v:
    valfile = v.readlines()
# ///////////////////////////////////////////////////////// #

# the training steps
if RESTORE:
    AO_model = load_model('./saved_models_AO/AOmodel-2p-001-0.00000.h5')
else:
    AO_model = AO.AO_model(people_num)

train_generator = AudioGenerator(trainfile,database_dir_path= '../../data/audio/audio_database', batch_size=batch_size, shuffle=True)
val_generator = AudioGenerator(valfile,database_dir_path='../../data/audio/audio_database', batch_size=batch_size, shuffle=True)

AO_model.fit_generator(generator=train_generator,
                       validation_data=val_generator,
                       epochs=epochs,
                       callbacks=[TensorBoard(log_dir='./log_AO'), checkpoint, rlr],
                       initial_epoch=initial_epoch
                       )



























