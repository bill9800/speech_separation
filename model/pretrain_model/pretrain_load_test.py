import tensorflow as tf
import sys
sys.path.append('../lib')
import utils
from tensorflow.python.framework import tensor_util
import numpy as np

data = np.random.randint(256,size=(1,160,160,3),dtype='int32')


##############
graph_path = 'FaceNet_new/20180402-114759.pb'

graph = utils.load_graph(graph_path,tensorboard=False)
# utils.inspect_operation(graph_path,'ops.txt')

x = graph.get_tensor_by_name('batch_size:0')
y = graph.get_tensor_by_name('phase_train:0')
z = graph.get_tensor_by_name('embeddings:0')

with tf.Session(graph=graph) as sess:
    print(sess.run(z,feed_dict={x:data,y:0}))

'''
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('FaceNet_old/model-20170512-110547.meta')
    saver.restore(sess,'FaceNet_old/model-20170512-110547.ckpt-250000.data-00000-of-00001')
    print(sess.run(global_step_tensor))
'''







