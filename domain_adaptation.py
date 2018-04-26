
from keras_helper import NNWeightHelper
from snes import SNES
from keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from sequencialsnes import SSNES



class DomainAdaptation():

    def __init__(self,source, target, feature_extractor):
        self.source_trainX=np.asarray(source[0][0])
        self.source_trainY=np.asarray(source[0][1])
        self.source_validationX=np.asarray(source[1][0])
        self.source_validationY=np.asarray(source[1][1])

        self.target_trainX = np.asarray(target[0][0])
        self.target_trainY = np.asarray(target[0][1])
        self.target_validationX = np.asarray(target[1][0])
        self.target_validationY = np.asarray(target[1][1])

        self.mixed_trainX=np.concatenate((self.source_trainX, self.target_trainX), axis=0)
        self.mixed_trainY=np.concatenate((np.zeros(self.source_trainX.shape[0]), np.ones(self.target_trainX.shape[0])),axis=0)

        self.mixed_validationX = np.concatenate((self.source_validationX, self.target_validationX),axis=0)
        self.mixed_validationY = np.concatenate((np.zeros(self.source_validationX.shape[0]), np.ones(self.target_validationX.shape[0])),axis=0)
        
        self.feature_extractor=feature_extractor

        


        self.makeModels()

    def makeModels(self):
        self.label_clf = RandomForestClassifier()
        self.domain_clf = RandomForestClassifier()

    def clf_train(self,model, x, y):
        x_features=model.predict(x)
        clf = RandomForestClassifier(n_estimators=18)
        clf=clf.fit(x_features, y)
        return clf

    def clf_predict(self,model, clf,x):
        x_features=model.predict(x)
        y=clf.predict(x_features)
        return y

    def saveModel(self,filename):
        file=open(filename+'.json','w')
        file.write(self.feature_extractor.to_json())
        self.feature_extractor.save_weights(filename+'.h5')
        file.close()

    def load_model(self,filename):
        file=open(filename+'.json')
        model=file.read()
        self.feature_extractor=model_from_json(model)
        self.feature_extractor.load_weights(filename+'.h5')

    def fitness(self,label_accuracy,domain_accuracy):
        fitness = label_accuracy**2 /10*domain_accuracy
        return fitness

    def invariant_loss(self, domain_ac):
        f= 1/domain_ac
        #f=1-domain_ac
        return 1-f







    def train(self):

        random_domainTrain_indices = np.random.choice(a=list(range(self.mixed_trainX.shape[0])),size=1024)
        mixed_domain_trainX = self.mixed_trainX[random_domainTrain_indices]
        mixed_domain_trainY = self.mixed_trainY[random_domainTrain_indices]

        random_domainValid_indices = np.random.choice(a=list(range(self.mixed_validationX.shape[0])), size=1024)
        mixed_domain_validX = self.mixed_validationX[random_domainValid_indices]
        mixed_domain_validY = self.mixed_validationY[random_domainValid_indices]

        source_validation_indices = np.random.choice(a=list((range(self.source_validationX.shape[0]))),size=1024)
        validX = self.source_validationX[source_validation_indices]
        validY = self.source_validationY[source_validation_indices]



        self.label_clf = self.clf_train(self.feature_extractor, self.source_trainX, self.source_trainY)
        label_pred = self.clf_predict(self.feature_extractor, self.label_clf, validX)
        label_accuracy = accuracy_score(validY, label_pred)

        self.domain_clf = self.clf_train(self.feature_extractor,mixed_domain_trainX, mixed_domain_trainY)
        domain_pred = self.clf_predict(self.feature_extractor,self.domain_clf, mixed_domain_validX)
        domain_accuracy = accuracy_score(mixed_domain_validY, domain_pred)

        print('Baseline label-clf accuracy: %0.3f, baseline domain_clf: %0.3f, domain_invariant label-accuracy , domain_accurcy ' %(label_accuracy, domain_accuracy))
        weight_modifier = NNWeightHelper(self.feature_extractor)
        weights = weight_modifier.get_weights()
        print('total weights to evolve:',len(weights) )

        snes=SSNES(weights,1,100)

        batchScores = []
        batch_clf_acc = []
        batch_domain_acc = []

        for i in range(20):

            fitnesses = []
            domain_accuracys = []
            label_accuracys = []
            indices = []

            for i in range(10):

                new_weights, index = snes.predict()
                weight_modifier.set_weights(new_weights)
                indices.append(index)


                self.label_clf=self.clf_train(self.feature_extractor,self.source_trainX[source_validation_indices],self.source_trainY[source_validation_indices])
                label_predictions = self.clf_predict(self.feature_extractor,self.label_clf, validX)
                label_accuracy=accuracy_score(validY,label_predictions)

                self.domain_clf = self.clf_train(self.feature_extractor,mixed_domain_trainX,mixed_domain_trainY)
                domain_predictions=self.clf_predict(self.feature_extractor,self.domain_clf,mixed_domain_validX)
                domain_accuracy=accuracy_score(mixed_domain_validY,domain_predictions)
                domain_f1 = f1_score(mixed_domain_validY, domain_pred)

                fitnesses.append(self.fitness(label_accuracy, domain_f1 ))
                domain_accuracys.append(domain_accuracy)
                label_accuracys.append(label_accuracy)

            # batchScores.append(fitnesses)
            
            batchScores.append(np.mean(fitnesses))
            batch_clf_acc.append(np.mean(label_accuracys))
            batch_domain_acc.append(np.mean(domain_accuracys))

            most_fit_model = np.argmax(fitnesses)
            print("Most fit model has label_accuracy: %0.3f and domain_accuracy:%0.3f and fitness_score:%0.3f" % (label_accuracys[most_fit_model], domain_accuracys[most_fit_model], fitnesses[most_fit_model]))
            snes.fit(fitnesses,indices)


        weight_modifier.set_weights(snes.snes.center)
        self.label_clf = self.clf_train(self.feature_extractor,self.source_trainX, self.source_trainY)

        random_domainTrain_indices = np.random.choice(a=list(range(self.mixed_trainX.shape[0])), size=self.mixed_trainX.shape[0])
        all_domain_trainX = self.mixed_trainX[random_domainTrain_indices]
        all_domain_trainY = self.mixed_trainY[random_domainTrain_indices]
        self.domain_clf = self.clf_train(self.feature_extractor,all_domain_trainX,all_domain_trainY)

        random_domainValid_indices = np.random.choice(a=list(range(self.mixed_validationX.shape[0])), size=self.mixed_validationX.shape[0])
        all_domain_testX = self.mixed_validationX[random_domainValid_indices]
        all_domain_testY = self.mixed_validationY[random_domainValid_indices]

        source_label_predictions=self.clf_predict(self.feature_extractor,self.label_clf,self.source_validationX)
        target_label_predictions=self.clf_predict(self.feature_extractor,self.label_clf,self.target_trainX)
        domain_predictions=self.clf_predict(self.feature_extractor,self.domain_clf,all_domain_testX)
        print('**-validation--**')
        print('label-predicitions on source data: %0.3f'%(accuracy_score(self.source_validationY, source_label_predictions)))
        print('label-predicitions on target data: %0.3f' % (accuracy_score(self.target_trainY, target_label_predictions)))
        print('domain-predicitions on source data: %0.3f' % (accuracy_score(all_domain_testY, domain_predictions)))