import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn import svm
from matplotlib import rcParams
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from matplotlib import cm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

class subtask_a:
    def __init__(self, data):
        self.data = data

    def data_explore(self):
        print(f"There are {len(self.data)} records in the csv file.")
        print(f"There are {len(self.data[0].keys())} attributes per row.\nThese are: {list(self.data[0].keys())}")
        df = pd.read_csv('dataset/Transformed_ODI.csv')
        df = df.apply(lambda x: x.astype(str).str.lower())

        df = df.replace('yes', 1)
        df = df.replace('no', 0)
        df = df.replace('mu', 1)
        df = df.replace('sigma', 0)
        df = df.replace('female', 1)
        df = df.replace('male', 0)
        #df.loc[df['What is your stress level (0-100)?']] = np.nan
        df = df.replace('unknown', np.nan)
        #df.loc[df['What is your stress level (0-100)?']>100] = np.nan
        #df.loc[df['What is your stress level (0-100)?']< 0] = np.nan

        yvar = df.iloc[:,11]
        xvar = df.iloc[:,2:5]

        #df2 = list(df.loc[:, 'What is your stress level (0-100)?'])+['Have you taken a course on machine learning?']

        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        df_full = pd.DataFrame(imp.fit_transform(xvar),columns=['Have you taken a course on machine learning?', 'Have you taken a course on information retrieval?', 'Have you taken a course on statistics'])
        df_inv = pd.DataFrame(imp.fit_transform(df.iloc[:,2:4]),columns=['Have you taken a course on machine learning?', 'Have you taken a course on information retrieval?'])
        df_3 = df.iloc[:,2]
        df_4 = pd.DataFrame(imp.fit_transform(df.iloc[:,[11,12]]))


        #X_train, X_test, y_train, y_test = train_test_split(df_full,yvar,random_state=0)
        #xCL_train, xCL_test, yCL_train, yCL_test = train_test_split(df_inv,yvar,random_state=0)
        x3_train,x3_test,y3_train,y3_test = train_test_split(df.iloc[:,11].values.reshape(-1,1),df_3,random_state=1)
        x4_train,x4_test,y4_train,y4_test = train_test_split(df_4,df_3,test_size=0.4,random_state=1)


        # scaler = MinMaxScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        # ###logistic regression algorithm
        # LR = LogisticRegression()
        # LR.fit(X_train,y_train)
        # print('Accuracy of Logistic regression classifier on training set: {:.4f}'
        #       .format(LR.score(X_train, y_train)))
        # print('Accuracy of Logistic regression classifier on test set: {:.4f}'
        #       .format(LR.score(X_test, y_test)))
        #
        # ###logistic regression algorithm cl2
        # LR = LogisticRegression()
        # LR.fit(xCL_train, yCL_train)
        # print('Accuracy of Logistic regression (w/o stat course) classifier on training set: {:.4f}'
        #       .format(LR.score(xCL_train, yCL_train)))
        # print('Accuracy of Logistic regression classifier on test set: {:.4f}'
        #       .format(LR.score(xCL_test, yCL_test)))

        ###Decision tree
        # clf = DecisionTreeClassifier().fit(X_train,y_train)
        # print('Accuracy of Decision Tree classifier on training set: {:.4f}'
        #       .format(clf.score(X_train, y_train)))
        # print('Accuracy of Decision Tree classifier on test set: {:.4f}'
        #       .format(clf.score(X_test, y_test)))

        clf = GridSearchCV(DecisionTreeClassifier(),{'max_depth':range(3,20)} , n_jobs=4)
        clf.fit(X=df_4, y=df_3)
        tree_model = clf.best_estimator_
        print(f"Cross validation results for Decision tree classifier:\nBest parameters: {clf.best_params_}, Best score: {clf.best_score_}")

        #print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        # clf = DecisionTreeClassifier().fit(x3_train.reshape(-1, 1), y3_train)
        # print('Accuracy of Decision Tree classifier on training set x3: {:.4f}'
        #       .format(clf.score(x3_train, y3_train)))
        # print('Accuracy of Decision Tree classifier on test setx3: {:.4f}'
        #       .format(clf.score(x3_test, y3_test)))
        #
        # clf = DecisionTreeClassifier().fit(x4_train, y4_train)
        # print('Accuracy of Decision Tree classifier on training set x4: {:.4f}'
        #       .format(clf.score(x4_train, y4_train)))
        print('Accuracy of Decision Tree classifier on test set x4: {:.4f}'
              .format(clf.score(x4_test, y4_test)))

        # ###Decision tree with inv data
        # clf = DecisionTreeClassifier().fit(xCL_train, yCL_train)
        # print('Accuracy of Decision Tree classifier (w/o stat course) on training set: {:.4f}'
        #       .format(clf.score(xCL_train, yCL_train)))
        # print('Accuracy of Decision Tree classifier on test set: {:.4f}'
        #       .format(clf.score(xCL_test, yCL_test)))


        ###K-nearest neighbours
        # knn = KNeighborsClassifier()
        # knn.fit(X_train,y_train)
        # print("K-NN accuracy on training set: {:.4f}".format(knn.score(X_train, y_train)))
        # print("K-NN accuracy on test set: {:.4f}".format(knn.score(X_test, y_test)))


        # knn = KNeighborsClassifier()
        # knn.fit(x3_train, y3_train)
        # print("K-NN accuracy on training set x3: {:.4f}".format(knn.score(x3_train, y3_train)))
        # print("K-NN accuracy on test set x3: {:.4f}".format(knn.score(x3_test, y3_test)))

        knn = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': range(3, 12)}, n_jobs=4)
        knn.fit(x4_train, y4_train)
        print(f"Cross validation results for K-nearest neighbors:\nBest parameter: {knn.best_params_}, best score: {knn.best_score_}")

        # print("K-NN accuracy on training set x4: {:.4f}".format(knn.score(x4_train, y4_train)))
        print("K-NN accuracy on test set x4: {:.4f}".format(knn.score(x4_test, y4_test)))

        # Does same thing as above, thus unnecessary
        # knn = KNeighborsClassifier(n_neighbors=10)
        # knn.fit(x4_train, y4_train)
        # print("K-NN accuracy on training set x4: {:.4f}".format(knn.score(x4_train, y4_train)))
        # print("K-NN accuracy on test set x4: {:.4f}".format(knn.score(x4_test, y4_test)))

        #cmap = cm.get_cmap('gnuplot')
        # scatter = scatter_matrix(df2,c=yvar, marker='o',s=40,hist_kwds={'bins':15},figsize=(9,9),cmap = cmap)


        #yVar = df.iloc[:,20]
        # df2 = df[xVar]
        #print(df['Have you taken a course on information retrieval?'][58:62])
        #print(df.isnull().any())
        #enc = OneHotEncoder(handle_unknown="ignore")
        #enc.fit(df['Have you taken a course on machine learning?'])
        #enc.categories_