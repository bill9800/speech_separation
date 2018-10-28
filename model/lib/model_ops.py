import tensorflow as tf
import os, argparse
from tensorflow.python.tools.freeze_graph import freeze_graph

def save_model(model_dir,output_name,type='pb'):
    if type == 'pb':
        freeze_graph(model_dir,output_name)


def load_graph(graph_path,tensorboard=False,**kwargs):
    '''
    :param graph_filename: the path of the pb file
    :return: tensorflow graph
    '''
    with gfile.FastGFile(graph_path,'rb') as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name="")

    if tensorboard:
        writer = tf.summary.FileWriter("log/")
        writer.add_graph(graph)

    return graph


def inspect_operation(graph_path,output_txt_file):
    '''
    :param graph_path: the path of the pb file
    :param output_txt_file: the path of the txt outputfile for inspect the model
    :return:
    '''
    graph = load_graph(graph_path)
    with tf.Session(graph=graph) as sess:
        operations = sess.graph.get_operations()

    ops_dict = {}
    with open(output_txt_file,'w') as f:
        for i, op in enumerate(operations):
            f.write('---------------------------------------------------------------------------------------------\n')
            f.write("{}: op name = {}, op type = ( {} ), inputs = {}, outputs = {}\n".\
                  format(i, op.name, op.type, ", ".join([x.name for x in op.inputs]), ", ".join([x.name for x in op.outputs])))
            f.write('@input shapes:\n')
            for x in op.inputs:
                f.write("name = {} : {}\n".format(x.name, x.get_shape()))
                f.write('@output shapes:\n')
            for x in op.outputs:
                f.write("name = {} : {}\n".format(x.name, x.get_shape()))
            if op.type in ops_dict:
                ops_dict[op.type] += 1
            else:
                ops_dict[op.type] = 1

                f.write('---------------------------------------------------------------------------------------------\n')
        sorted_ops_count = sorted(ops_dict.items(), key=operator.itemgetter(1))
        print('OPS counts:')
        for i in sorted_ops_count:
            print("{} : {}".format(i[0], i[1]))


