import tensorflow.compat.v1 as tf
import numpy as np
from skimage.transform import resize
import os

PB_FILE = os.path.join(os.path.dirname(__file__), 'pretrained_weights', 'brainextractor.pb')

class Extractor:

    def __init__(self):
        self.SIZE = 128
        self.load_pb()

    def load_pb(self):
        graph = tf.Graph()
        self.sess = tf.Session(graph=graph)
        with tf.gfile.FastGFile(PB_FILE, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            with self.sess.graph.as_default():
                tf.import_graph_def(graph_def)

        self.img = graph.get_tensor_by_name("import/img:0")
        self.training = graph.get_tensor_by_name("import/training:0")
        self.dim = graph.get_tensor_by_name("import/dim:0")
        self.prob = graph.get_tensor_by_name("import/prob:0")
        self.pred = graph.get_tensor_by_name("import/pred:0")

    def run(self, image):
        shape = image.shape
        img = resize(image, (self.SIZE, self.SIZE, self.SIZE), mode='constant', anti_aliasing=True)
        img = (img / np.max(img))
        img = np.reshape(img, [1, self.SIZE, self.SIZE, self.SIZE, 1])

        prob = self.sess.run(self.prob, feed_dict={self.training: False, self.img: img}).squeeze()
        prob = resize(prob, (shape), mode='constant', anti_aliasing=True)
        return prob


