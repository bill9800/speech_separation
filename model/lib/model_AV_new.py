from keras import optimizers
#from keras.layers import Dense, Convolution3D, MaxPooling3D, ZeroPadding3D, Dropout, Flatten, BatchNormalization, ReLU
from keras.models import Sequential, model_from_json
from keras import optimizers
from keras.layers import Input, Dense, Convolution2D, Deconvolution2D, Bidirectional, UpSampling2D, UpSampling3D, concatenate
from keras.layers import Dropout, Flatten, BatchNormalization, ReLU, Reshape, Permute, Lambda, TimeDistributed
from keras.models import Model, load_model
from keras.layers.recurrent import LSTM
from keras.initializers import he_normal, glorot_uniform
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import TensorBoard
import tensorflow as tf
import os
from MyGenerator import AVGenerator


def AV_model(people_num=2):
    def UpSampling2DBilinear(size):
        return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))

    def sliced(x, index):
        return x[..., index]

    # --------------------------- AS start ---------------------------
    audio_input = Input(shape=(298, 257, 2))
    print('as_0:', audio_input.shape)
    as_conv1 = Convolution2D(96, kernel_size=(1, 7), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv1')(audio_input)
    as_conv1 = BatchNormalization()(as_conv1)
    as_conv1 = ReLU()(as_conv1)
    print('as_1:', as_conv1.shape)

    as_conv2 = Convolution2D(96, kernel_size=(7, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv2')(as_conv1)
    as_conv2 = BatchNormalization()(as_conv2)
    as_conv2 = ReLU()(as_conv2)
    print('as_2:', as_conv2.shape)

    as_conv3 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv3')(as_conv2)
    as_conv3 = BatchNormalization()(as_conv3)
    as_conv3 = ReLU()(as_conv3)
    print('as_3:', as_conv3.shape)

    as_conv4 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 1), name='as_conv4')(as_conv3)
    as_conv4 = BatchNormalization()(as_conv4)
    as_conv4 = ReLU()(as_conv4)
    print('as_4:', as_conv4.shape)

    as_conv5 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 1), name='as_conv5')(as_conv4)
    as_conv5 = BatchNormalization()(as_conv5)
    as_conv5 = ReLU()(as_conv5)
    print('as_5:', as_conv5.shape)

    as_conv6 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 1), name='as_conv6')(as_conv5)
    as_conv6 = BatchNormalization()(as_conv6)
    as_conv6 = ReLU()(as_conv6)
    print('as_6:', as_conv6.shape)

    as_conv7 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 1), name='as_conv7')(as_conv6)
    as_conv7 = BatchNormalization()(as_conv7)
    as_conv7 = ReLU()(as_conv7)
    print('as_7:', as_conv7.shape)

    as_conv8 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 1), name='as_conv8')(as_conv7)
    as_conv8 = BatchNormalization()(as_conv8)
    as_conv8 = ReLU()(as_conv8)
    print('as_8:', as_conv8.shape)

    as_conv9 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv9')(as_conv8)
    as_conv9 = BatchNormalization()(as_conv9)
    as_conv9 = ReLU()(as_conv9)
    print('as_9:', as_conv9.shape)

    as_conv10 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 2), name='as_conv10')(as_conv9)
    as_conv10 = BatchNormalization()(as_conv10)
    as_conv10 = ReLU()(as_conv10)
    print('as_10:', as_conv10.shape)

    as_conv11 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 4), name='as_conv11')(as_conv10)
    as_conv11 = BatchNormalization()(as_conv11)
    as_conv11 = ReLU()(as_conv11)
    print('as_11:', as_conv11.shape)

    as_conv12 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 8), name='as_conv12')(as_conv11)
    as_conv12 = BatchNormalization()(as_conv12)
    as_conv12 = ReLU()(as_conv12)
    print('as_12:', as_conv12.shape)

    as_conv13 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 16), name='as_conv13')(as_conv12)
    as_conv13 = BatchNormalization()(as_conv13)
    as_conv13 = ReLU()(as_conv13)
    print('as_13:', as_conv13.shape)

    as_conv14 = Convolution2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 32), name='as_conv14')(as_conv13)
    as_conv14 = BatchNormalization()(as_conv14)
    as_conv14 = ReLU()(as_conv14)
    print('as_14:', as_conv14.shape)

    as_conv15 = Convolution2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv15')(as_conv14)
    as_conv15 = BatchNormalization()(as_conv15)
    as_conv15 = ReLU()(as_conv15)
    print('as_15:', as_conv15.shape)

    AS_out = Reshape((298, 8 * 257))(as_conv15)
    print('AS_out:', AS_out.shape)
    # --------------------------- AS end ---------------------------

    # --------------------------- VS_model start ---------------------------
    VS_model = Sequential()
    VS_model.add(Convolution2D(256, kernel_size=(7, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='vs_conv1'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='vs_conv2'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(2, 1), name='vs_conv3'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(4, 1), name='vs_conv4'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(8, 1), name='vs_conv5'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(16, 1), name='vs_conv6'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Reshape((75, 256, 1)))
    VS_model.add(UpSampling2DBilinear((298, 256)))
    VS_model.add(Reshape((298, 256)))
    # --------------------------- VS_model end ---------------------------

    video_input = Input(shape=(75, 1, 1792, people_num))
    AVfusion_list = [AS_out]
    for i in range(people_num):
        single_input = Lambda(sliced, arguments={'index': i})(video_input)
        VS_out = VS_model(single_input)
        AVfusion_list.append(VS_out)

    AVfusion = concatenate(AVfusion_list, axis=2)
    AVfusion = TimeDistributed(Flatten())(AVfusion)
    print('AVfusion:', AVfusion.shape)

    lstm = Bidirectional(LSTM(400, input_shape=(298, 8 * 257), return_sequences=True), merge_mode='sum')(AVfusion)
    print('lstm:', lstm.shape)

    fc1 = Dense(600, name="fc1", activation='relu', kernel_initializer=he_normal(seed=27))(lstm)
    print('fc1:', fc1.shape)
    fc2 = Dense(600, name="fc2", activation='relu', kernel_initializer=he_normal(seed=42))(fc1)
    print('fc2:', fc2.shape)
    fc3 = Dense(600, name="fc3", activation='relu', kernel_initializer=he_normal(seed=65))(fc2)
    print('fc3:', fc3.shape)

    complex_mask = Dense(257 * 2 * people_num, name="complex_mask", kernel_initializer=glorot_uniform(seed=87))(fc3)
    print('complex_mask:', complex_mask.shape)

    complex_mask_out = Reshape((298, 257, 2, people_num))(complex_mask)
    print('complex_mask_out:', complex_mask_out.shape)

    AV_model = Model(inputs=[audio_input, video_input], outputs=complex_mask_out)

    # # compile AV_model
    # AV_model.compile(optimizer='adam', loss='mse')

    return AV_model


if __name__ == '__main__':
    #############################################################
    RESTORE = False
    # If set true, continue training from last checkpoint
    # needed change 1:h5 file name, 2:epochs num, 3:initial_epoch

    # super parameters
    people_num = 2
    epochs = 50
    initial_epoch = 0
    batch_size = 2
    #############################################################

    # audio_input = np.random.rand(5, 298, 257, 2)        # 5 audio parts, (298, 257, 2) stft feature
    # audio_label = np.random.rand(5, 298, 257, 2, people_num)     # 5 audio parts, (298, 257, 2) stft feature, people num to be defined

    # ///////////////////////////////////////////////////////// #
    # create folder to save models
    path = './saved_models_AV'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('create folder to save models')
    filepath = path + "/AVmodel-" + str(people_num) + "p-{epoch:03d}-{val_loss:.10f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


    # checkpoint2 = ModelCheckpoint(path + "/AOmodel-latest-" + str(people_num) + ".h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # ///////////////////////////////////////////////////////// #

    #############################################################
    # automatically change lr
    def scheduler(epoch):
        ini_lr = 0.001
        lr = ini_lr
        if epoch >= 5:
            lr = ini_lr / 5
        if epoch >= 10:
            lr = ini_lr / 10
        return lr


    rlr = LearningRateScheduler(scheduler, verbose=1)
    #############################################################

    # ///////////////////////////////////////////////////////// #
    # read train and val file name
    # format: mix.npy single.npy single.npy
    trainfile = []
    valfile = []
    with open('./trainfile.txt', 'r') as t:
        trainfile = t.readlines()
    with open('./valfile.txt', 'r') as v:
        valfile = v.readlines()


    # ///////////////////////////////////////////////////////// #

    # the training steps
    def latest_file(dir):
        lists = os.listdir(dir)
        lists.sort(key=lambda fn: os.path.getmtime(dir + fn))
        file_latest = os.path.join(dir, lists[-1])
        return file_latest


    if RESTORE:
        last_file = latest_file('./saved_models_AO/')
        AV_model = load_model(last_file)
        info = last_file.strip().split('-')
        initial_epoch = int(info[-2])
        # print(initial_epoch)
    else:
        AV_model = AV_model(people_num)
        adam = optimizers.Adam()
        AV_model.compile(optimizer=adam, loss='mse')

    train_generator = AVGenerator(trainfile, database_dir_path='./', batch_size=batch_size, shuffle=True)
    val_generator = AVGenerator(valfile, database_dir_path='./', batch_size=batch_size, shuffle=True)

    AV_model.fit_generator(generator=train_generator,
                           validation_data=val_generator,
                           epochs=epochs,
                           callbacks=[TensorBoard(log_dir='./log_AV'), checkpoint, rlr],
                           initial_epoch=initial_epoch
                           )
