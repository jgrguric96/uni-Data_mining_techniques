import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import random


df = pd.read_csv('Data/training_set_VU_DM.csv')
# df = full[:10000]
#y_query = pd.read_csv('Data/test_set_VU_DM.csv')

mu, sigma = 5.030700, 0.524022  # mean and standard deviation. We got it from rstudio
s = np.random.lognormal(mu, sigma, 4958347) #the value 4958347 is literally just the length of the DF. we can also get this with .len instead of a hardcoded value
VHAU_column = pd.DataFrame({'visitor_hist_adr_usd':s})

df.iloc[df.price_usd > 5000, [15]] = 5000
df['orig_destination_distance'] = df['orig_destination_distance'].fillna(df['orig_destination_distance'].mean())

df['prop_starrating'] = df['prop_starrating'].astype(float, copy=False)

df['prop_review_score'] = df['prop_review_score'].fillna(0)
df['prop_review_score'] = df['prop_review_score'].astype(float, copy=False)

#Likely will need to be changed
df['visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(df['visitor_hist_starrating'].mean())

df = df.assign(visitor_hist_adr_usd=VHAU_column['visitor_hist_adr_usd'])

#Could be changed
# df['prop_location_score1'] = pd.cut(df['prop_location_score1'],
#                   bins=[0, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.5],
#                   include_lowest=True,
#                   labels=['0', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5', '5.5', '6', '6.5', '7.0'])
df['prop_location_score1'] = df['prop_location_score1'].astype(float, copy=False)

df['srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(0)
# df['srch_query_affinity_score'] = pd.cut(df['srch_query_affinity_score'],
#                                           bins=[-1000, -100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50,
#                                                 -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 100],
#                                           include_lowest=True,
#                                           labels=["100 point", "95 point", "90 point", "85 point", "80 point",
#                                                   "75 point", "70 point", "65 point", "60 point", "55 point",
#                                                   "50 point", "45 point", "40 point", "35 point", "30 point",
#                                                   "25 point", "20 point", "15 point", "10 point", "5 point",
#                                                   "0 point", 'Not Applicable'])
df['srch_query_affinity_score'] = df['srch_query_affinity_score'].astype(float, copy=False)

# df['prop_location_score2'] = df['prop_location_score2'].fillna(0)
# df['prop_location_score2'] = pd.cut(df['prop_location_score2'],
#                                           bins=20,
#                                           include_lowest=True)
# df['prop_location_score2'] = df['prop_location_score2'].astype(float, copy=False)

df['assign_score'] = df.click_bool + 4*df.booking_bool
# df['date_time'] = pd.to_datetime(df['date_time'])
# df['year'] = df['date_time'].dt.year
# df['month'] = df['date_time'].dt.month

comp_pct = [ 'comp1_rate_percent_diff',
       'comp2_rate_percent_diff',
       'comp3_rate_percent_diff',
       'comp4_rate_percent_diff',
       'comp5_rate_percent_diff',
       'comp6_rate_percent_diff',
       'comp7_rate_percent_diff',
       'comp8_rate_percent_diff']

comp_rates = ['comp1_rate',
'comp2_rate',
'comp3_rate',
'comp4_rate',
'comp5_rate',
'comp6_rate',
'comp7_rate',
'comp8_rate']

comps = pd.DataFrame()
for i, comp in enumerate(comp_rates):
  comps[str(i)] = df[comp] * df[comp_pct[i]]

comps = comps.fillna(0)

df['best_competitor'] = comps.min(axis=1) #negative values represent a competitor with a better price than Expedia in %
df['worst_competitor'] = comps.max(axis=1) #positive values represent a competitor with a worse price than Expedia in %

df['best_competitor'] = df['best_competitor'].astype(float, copy=False)
df['worst_competitor'] = df['worst_competitor'].astype(float, copy=False)


# print(df.dtypes)
# df.to_csv("training_set_XGBoost.csv")

df['prop_location_score1'].to_csv('prop_loc_1.csv')
df['prop_location_score2'].to_csv('prop_loc_2.csv')



# test_columns = df.drop(['comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff',
#        'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff',
#        'comp3_rate', 'comp3_inv',
#        'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',
#        'comp4_rate_percent_diff', 'comp5_rate',
#        'comp5_inv', 'comp5_rate_percent_diff',
#        'comp6_rate',
#        'comp6_inv',
#        'comp6_rate_percent_diff', 'comp7_rate',
#        'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate',
#        'comp8_inv', 'comp8_rate_percent_diff'], axis=1)
# print(test_columns.head())
# predictors = [c for c in test_columns.columns if c not in ["click_bool"]]
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
# print(clf.get_params())


# unique_users = set(df.srch_id.unique())
# rand_user_id = random.sample(unique_users,10000)
#
# sample_train = pd.DataFrame()
# train_chunk = pd.read_csv('Data/preprocessed_df.csv', iterator = True, chunksize = 1000000)
# for chunk in train_chunk:
#     sample_train = sample_train.append(chunk.loc[chunk['srch_id'].isin(rand_user_id)])
#
# sample_train['date_time'] = pd.to_datetime(sample_train['date_time'])
# sample_train['year'] = sample_train['date_time'].dt.year
# sample_train['month'] = sample_train['date_time'].dt.month
#
# sample_train.info()
# train_sub, test_sub = train_test_split(sample_train, test_size=0.3)
# test_sub = test_sub[test_sub.click_bool == 1]
#
# most_common_clusters = list(df.srch_id.value_counts().head().index)
# print(most_common_clusters)
# predictions = [most_common_clusters for i in range(test_sub.shape[0])]
#
# import ml_metrics as metrics
# target = [[l] for l in test_sub['prop_id']]
# metrics.mapk(target, predictions, k=5)
# df = train_sub.drop(['position', 'click_bool', 'booking_bool', 'gross_bookings_usd', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff',
#        'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff',
#        'comp3_rate', 'comp3_inv',
#        'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',
#        'comp4_rate_percent_diff', 'comp5_rate',
#        'comp5_inv', 'comp5_rate_percent_diff',
#        'comp6_rate',
#        'comp6_inv',
#        'comp6_rate_percent_diff', 'comp7_rate',
#        'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate',
#        'comp8_inv', 'comp8_rate_percent_diff', 'date_time', 'prop_location_score2','gross_bookings_usd', 'srch_query_affinity_score'], axis=1)
#
# df.fillna(-1, inplace=True)
# predictors = [c for c in df.columns if c not in ["prop_id"]]
# predictors = ['prop_country_id', 'prop_review_score', 'prop_location_score2']
# df['prop_location_score2'] = df['prop_location_score2'].fillna(df['prop_location_score2'].mean())
# df['prop_review_score'] = df['prop_review_score'].fillna(df['prop_review_score'].mean())
#
# X = df[predictors]
# y = df['prop_id'].astype('category', copy=False)
# X['prop_country_id'] = X['prop_country_id'].astype(int, copy=False)
# X['prop_review_score'] = X['prop_review_score'].astype(float, copy=False)
# # X['price_usd'] = X['price_usd'].astype(float, copy=False)
# X['prop_location_score2'] = X['prop_location_score2'].astype(float, copy=False)
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# print("train-test_split_done")
#
# # #Linear regression
# # reg = LinearRegression().fit(X_train, y_train)
# # preds = reg.predict(X_test)
# # print('Coefficients: \n', reg.coef_)
# # # The mean squared error
# # print('Mean squared error: %.2f'
# #       % mean_squared_error(y_test, preds))
# # # The coefficient of determination: 1 is perfect prediction
# # print('Coefficient of determination: %.2f'
# #       % r2_score(y_test, preds))
# #
# #
# # plt.scatter(X_test, y_test,  color='black')
# # plt.plot(X_test, preds, color='blue', linewidth=3)
# #
# # plt.xticks(())
# # plt.yticks(())
# #
# # plt.show()
#
# #K nearest
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print(f"Predicted with accuracy {accuracy_score(y_test, y_pred)}")
#
#
# #Decision tree
# from sklearn.tree import DecisionTreeClassifier
# tree = DecisionTreeClassifier(max_depth=10).fit(X_train,y_train)
# preds = tree.predict(X_test)
# print(f"Predicted with accuracy {accuracy_score(y_test, preds)}")


#MLP classifier
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 2), random_state=1)
#
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(f"Predicted NN with acc {accuracy_score(y_test, y_pred)}")

# #Grid Random Forest
# rf_params = {
#     "criterion": ['gini','entropy'],
#     "n_estimators": [10,50,100,200],
#     "max_features": ['auto','log2',None]
# }
# rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=3, verbose=10)
# rf_grid.fit(X_train, y_train)
# y_pred_test = rf_grid.predict(X_test)


# #Simple Random Forest
# clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
# #model = clf.fit(df[predictors], df['prop_id'])
# model2 = clf.fit(X_train, y_train)
#
# print(model2)
#
#
# print('data splitted')
#
# y_pred_test = clf.predict(X_test)
# print('accuracy')
# print(accuracy_score(y_test, y_pred_test))
# #print(classification_report(y_test, y_pred_test))




# #TODO K Means
# #KMean algorithm
# df = pd.read_csv('Data/preprocessed_df.csv')
#
# train=df.drop(["srch_id","date_time","site_id","visitor_location_country_id","prop_country_id","prop_id","srch_destination_id"],axis=1)
#
# train=train[['prop_starrating', 'prop_brand_bool', 'prop_location_score1',
#        'prop_log_historical_price', 'price_usd', 'promotion_flag',
#        'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
#        'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
#        'random_bool']]
#
# data = train
# n_cluster = range(1, 20)
#
# kmeans = [KMeans(n_clusters = i).fit(data) for i in n_cluster]
# scores = [kmeans[i].score(data) for i in range(len(kmeans))]
#
# from sklearn.cluster import KMeans
# data = train
# n_cluster = range(1, 20)
#
# print("Clustering done")
# kmeans = [KMeans(n_clusters = i).fit(data) for i in n_cluster]
# scores = [kmeans[i].score(data) for i in range(len(kmeans))]
#
# print(scores)