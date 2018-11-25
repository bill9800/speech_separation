import tensorflow as tf
import sys
sys.path.append('../lib')
import utils
from tensorflow.python.framework import tensor_util
import numpy as np
from keras.models import load_model
from keras.models import Model
## paraeter
PB = 0
CKPT = 0
HDF5 = 1

data = np.random.randint(256,size=(1,160,160,3),dtype='int32')


###############
graph_path = 'FaceNet_new/20180402-114759.pb'

# utils.inspect_operation(graph_path,'ops.txt')
if PB:
    with tf.gfile.FastGFile(graph_path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        with tf.Session() as sess:
            #sess.graph.as_default()
            tf.import_graph_def(graph_def)
            print(sess.run('import/embeddings', feed_dict={'import/batch_size:0': data,'import/phase_train:0':False}))

if CKPT:
    saver = tf.train.import_meta_graph('FaceNet_new/model-20180402-114759.meta')
    with tf.Session() as sess:
        saver.restore(sess=sess,save_path='FaceNet_new/model-20180402-114759.ckpt-275')
        #print(sess.run('embeddings:0', feed_dict={'batch_size:0': data, 'phase_train:0': False}))

if HDF5:
    model = load_model('FaceNet_keras/facenet_keras.h5')
    model.summary()
    avgPool_layer_model = Model(inputs=model.input,outputs=model.get_layer('AvgPool').output)
    print(avgPool_layer_model.predict(data))



