import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
import numpy as np


class t2s1:
    def __init__(self, survivorpath, trainpath, testpath):
        self.survivor = pd.read_csv(survivorpath)
        self.train = pd.read_csv(trainpath)
        self.test = pd.read_csv(testpath)

    def titanic(self):
        # print(self.survivor.describe())
        # print(self.train.describe())
        # print(self.test.describe())

        df = self.train.fillna(value=np.nan)
        df_test = self.test.fillna(value=np.nan)

        df = df.replace('male', 0)
        df = df.replace('female', 1)
        df_test = df_test.replace('male', 0)
        df_test = df_test.replace('female', 1)

        # mask = df.Cabin.where(df)
        # column_name = 'Cabin'
        # df.loc[mask, column_name] = 0



        #For Age values
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        X_svm = pd.DataFrame(imp.fit_transform(df.iloc[:, [2,4,5,6,7]]), columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
        X_test_svm = pd.DataFrame(imp.fit_transform(df_test.iloc[:, [1,3,4,5,6]]), columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
        y_svm = df.iloc[:, 1]


        #Base values for KNN and Rnadom forest classifier
        y = self.train["Survived"]
        y = y.astype('int')
        features = ["Pclass", "Sex", "SibSp", "Parch"]
        X = pd.get_dummies(self.train[features])
        X_test = pd.get_dummies(self.test[features])
        # print(X.shape, y.shape, X_test.shape)

        #Testing classifiers
        X_train, X2_test, y_train, y_test = train_test_split(X_svm, y_svm, random_state=1)

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        model.fit(X_train, y_train)
        print("Random forest: ", model.score(X2_test, y_test))

        model = KNeighborsClassifier(n_neighbors=10, leaf_size=50)
        model.fit(X_train, y_train)
        print("K_neighbours: ", model.score(X2_test, y_test))

        model = DecisionTreeClassifier(random_state=1)
        model.fit(X_train, y_train)
        print("Decision tree: ", model.score(X2_test, y_test))

        model = svm.LinearSVC(max_iter=1000, dual=False)
        model.fit(X_train, y_train)
        print("Linear svc: ", model.score(X2_test, y_test))

        model = GaussianProcessClassifier(kernel=1.0*RBF(1.0),random_state=1)
        model.fit(X, y)
        gaus_pred = model.predict(X_test)
        # print("Gaussian Classifier: ", model.score(X2_test, y_test))

        model = MLPClassifier(alpha=1, max_iter=1000,random_state=1)
        model.fit(X_train, y_train)
        print("Nerual Net Classifier: ", model.score(X2_test, y_test))



        output_gaus = pd.DataFrame({'PassengerId': self.test.PassengerId, 'Survived': gaus_pred})
        output_gaus.to_csv('gaus_prediction.csv', index=False)





        #Below this point only models being trained
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        model.fit(X, y)
        predictions = model.predict(X_test)

        knn = KNeighborsClassifier()
        knn.fit(X, y)
        knn_predict = knn.predict(X_test)

        SVM = svm.LinearSVC(max_iter=10000, dual=False)
        SVM.fit(X_svm, y_svm)
        svm_predict = SVM.predict(X_test_svm)

        #This one contains age. currently best with 77.9%
        forest_2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
        forest_2.fit(X_svm, y_svm)
        forest_predict = forest_2.predict(X_test_svm)
        # print(forest_2.score(X_svm, y_svm), 4)

        clf_dt = DecisionTreeClassifier(max_depth=5)
        clf_dt.fit(X_svm,y_svm)
        clf_predict = clf_dt.predict(X_test_svm)

        #Neural net
        nnet = MLPClassifier(alpha=1, max_iter=1000, random_state=1)
        nnet.fit(X_svm, y_svm)
        nnet_predict = nnet.predict(X_test_svm)


        output_forest = pd.DataFrame({'PassengerId': self.test.PassengerId, 'Survived': predictions})
        output_forest.to_csv('forest_prediction.csv', index=False)
        output_forest = pd.DataFrame({'PassengerId': self.test.PassengerId, 'Survived': forest_predict})
        output_forest.to_csv('forest2_prediction.csv', index=False)

        output_knn = pd.DataFrame({'PassengerId': self.test.PassengerId, 'Survived': knn_predict})
        output_knn.to_csv('knn_prediction.csv', index=False)

        output_svm = pd.DataFrame({'PassengerId': self.test.PassengerId, 'Survived': svm_predict})
        output_svm.to_csv('svm_prediction.csv', index=False)

        output_clf = pd.DataFrame({'PassengerId': self.test.PassengerId, 'Survived': clf_predict})
        output_clf.to_csv('clf_prediction.csv', index=False)

        output_nnet = pd.DataFrame({'PassengerId': self.test.PassengerId, 'Survived': nnet_predict})
        output_nnet.to_csv('nnet_prediction.csv', index=False)




