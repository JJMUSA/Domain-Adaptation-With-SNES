import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pickle as pkl
from domain_adaptation import DomainAdaptation
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,MaxPool2D

mnist=input_data.read_data_sets('MNIST', one_hot=True)

mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)

mnist_valid = (mnist.validation.images>0).reshape(mnist.validation.images.shape[0],28,28,1).astype(np.uint8)*255
mnist_valid=np.concatenate([mnist_valid,mnist_valid,mnist_valid],axis=3)
mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)



mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
mnistm_train = mnistm['train'].astype(np.uint8)* 255
mnistm_test = mnistm['test'].astype(np.uint8)* 255
mnistm_valid = mnistm['valid'].astype(np.uint8)* 255


#results are better with normalising
mnist_train = mnist_train.astype('float32')
mnist_valid = mnist_valid.astype('float32')
mnist_test = mnist_test.astype('float32')
mnist_max_pixel = np.amax(mnist_train)
mnist_train *=  (255/ mnist_max_pixel  )
mnist_valid *=  (255 /mnist_max_pixel )
mnist_test *=  (255 / mnist_max_pixel )

mnistm_train = mnistm_train.astype('float32')
mnistm_valid = mnistm_valid.astype('float32')
mnistm_test = mnistm_test.astype('float32')
mnistm_max_pixel = np.amax(mnistm_train)
mnistm_train *=  (255 / mnistm_max_pixel )
mnistm_valid *=  (255 / mnistm_max_pixel)
mnistm_test *=  (255 / mnistm_max_pixel)


train_labels = np.asarray([np.argwhere(i)[0][0] for i in mnist.train.labels])
valid_labels = np.asarray([np.argwhere(i)[0][0] for i in mnist.validation.labels])
test_labels = np.asarray([np.argwhere(i)[0][0] for i in mnist.test.labels])

source=[(mnistm_train, train_labels),(mnistm_valid, valid_labels)]
target=[(mnist_train, train_labels),(mnist_valid, valid_labels)]

input_shape = mnist_train.shape[1:]



feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, activation='relu', kernel_size= (5,5),input_shape = input_shape) )
feature_extractor.add(MaxPool2D())
feature_extractor.add(Conv2D(48,activation='relu',kernel_size=(3,3)))
feature_extractor.add(MaxPool2D())
feature_extractor.add(Flatten())
feature_extractor.add(Dense(15,activation='relu'))
feature_extractor.compile(loss='mse',optimizer='adam')

mnist_da=DomainAdaptation(source, target, feature_extractor)
# mnist_da.load_model('mnist-adaptation')
clf = mnist_da.clf_train(mnist_da.feature_extractor,mnist_da.source_trainX, mnist_da.source_trainY)
pred = mnist_da.clf_predict(mnist_da.feature_extractor, clf, mnistm_test)
print(pred)
base_targetAcc = accuracy_score(test_labels, pred )
print("Accurcy of untrained model on target set: %0.3f"%(base_targetAcc))
mnist_da.train()
# mnist_da.saveModel('mnist-adaptation3')

#testing
mnist_da.load_model('mnist-adaptation')
label_clf = mnist_da.clf_train(mnist_da.feature_extractor,mnist_train,train_labels)
# performance on source_data
source_predictions = mnist_da.clf_predict(mnist_da.feature_extractor,label_clf, mnist_test)
source_accuray = accuracy_score(test_labels,source_predictions)

# performance on target_data
target_predictions = mnist_da.clf_predict(mnist_da.feature_extractor,label_clf, mnistm_test)
target_accuracy = accuracy_score(target_predictions, test_labels)

# mixing source and target data
random_train_indices = np.random.choice(a=list(range(mnist_da.mixed_trainX.shape[0])), size=mnist_da.mixed_trainX.shape[0])
train_mixed_dataX = mnist_da.mixed_trainX[random_train_indices]
train_mixed_dataY = mnist_da.mixed_trainY[random_train_indices]

test_mixed_dataX = np.concatenate([mnist_test, mnistm_test], axis=0)
test_mixed_dataY = np.concatenate([np.zeros(mnistm_test.shape[0]), np.ones(mnist_test.shape[0])], axis=0)
random_test_indices = np.random.choice(a=list(range(test_mixed_dataX.shape[0])), size=test_mixed_dataX.shape[0])
test_mixed_dataX = test_mixed_dataX[random_test_indices]
test_mixed_dataY = test_mixed_dataY[random_test_indices]

domain_clf = mnist_da.clf_train(mnist_da.feature_extractor,train_mixed_dataX, train_mixed_dataY)
domain_pred = mnist_da.clf_predict(mnist_da.feature_extractor,domain_clf, test_mixed_dataX)
domain_accuracy= accuracy_score(test_mixed_dataY, domain_pred)
print('--Testing--')
print("label predicitions on source: %0.3f" % (source_accuray))
print("label predicitons on target: %0.3f" %(target_accuracy))
print("domain predicitions accuracy: %0.3f" %(domain_accuracy))



