## evaluate the model and generate the prediction

import sys
sys.path.append('../lib')
from keras.models import load_model
from MyGenerator import AudioGenerator

# load model
model_path = './saved_models_AO/AOmodel-2p-001-0.00000.h5'
AO_model = load_model(model_path)

# load data
testfile = []
with open('../../data/audio/audio_database/dataset_test.txt', 'r') as f:
    trainfile = f.readlines()

test_generator = AudioGenerator(testfile,database_dir_path= '../../data/audio/audio_database',
                                batch_size=2, shuffle=False)

# predict data
pred = AO_model.predict_generator(test_generator)
















