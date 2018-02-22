import pickle as pkl
from utils import *
from sklearn.manifold import TSNE


from tensorflow.examples.tutorials.mnist import input_data


mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
mnist_train=(mnist.train.images>0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)



mnistm=pkl.load(open("CE888/mnistm_data.pkl","rb")) #[train:55000,test:10000,valid:5000]
mnistm_train=mnistm["train"]
mnistm_test=mnistm["test"]
mnistm_valid=mnistm["valid"]
print(mnistm_train[0][-1].shape)
#imshow_grid(mnist_train)
