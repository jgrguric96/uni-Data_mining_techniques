import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class readData:
    def __init__(self, path_train, path_test):
        self.train_data = pd.read_csv(path_train, header=0)
        self.test_data = pd.read_csv(path_test, header=0)
        self.target_header = ["srch_id","prop_id"]

    def basic_data_info(self):
        # print(self.train_data.head())
        # print("Basic info on test data:\n", self.test_data.head())
        # corrmat = self.train_data.corr()
        # sns.heatmap(corrmat, square=True, annot=True)
        # train_x, test_x, train_y, test_y = self.__split_dataset__(self.train_data, 0.33, ["visitor_location_country_id"], self.target_header)
        # df = pd.DataFrame(self.train_data.iloc[:,15], columns=["price_usd"]
        # df = self.train_data

        mu, sigma = 5.030700, 0.524022  # mean and standard deviation

        s = np.random.lognormal(mu, sigma, 4958347)
        plt.hist(s, 200, density=True, align='mid')
        plt.show()
        df = pd.DataFrame({'visitor_hist_adr_usd':s})

        df_full = self.train_data


        plt.hist(df_full.visitor_hist_adr_usd, 150, density=True, align='mid')
        plt.show()

        df_full.iloc[df_full.price_usd > 5000, [15]] = 5000
        df_full['orig_destination_distance'] = df_full['orig_destination_distance'].fillna(df_full['orig_destination_distance'].mean())

        df_full['prop_starrating'] = df_full['prop_starrating'].astype('category', copy=False)

        df_full['prop_review_score'] = df_full['prop_review_score'].fillna(0)
        df_full['prop_review_score'] = df_full['prop_review_score'].astype('category', copy=False)

        #Really questionable
        df_full['visitor_hist_starrating'] = df_full['visitor_hist_starrating'].fillna(df_full['visitor_hist_starrating'].mean())

        df_full = df_full.assign(visitor_hist_adr_usd=df['visitor_hist_adr_usd'])

        #Questionable changes


        df_full['prop_location_score1'] = pd.cut(df_full['prop_location_score1'],
                         bins=[0, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.5],
                         include_lowest=True,
                         labels=['0', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5', '5.5', '6', '6.5', '7.0'])
        df_full['prop_location_score1'] = df_full['prop_location_score1'].astype('category', copy=False)

        df_full['srch_query_affinity_score'] = df_full['srch_query_affinity_score'].fillna(50)
        df_full['srch_query_affinity_score'] = pd.cut(df_full['srch_query_affinity_score'],
                                                 bins=[-1000, -100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50,
                                                       -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 100],
                                                 include_lowest=True,
                                                 labels=["100 point", "95 point", "90 point", "85 point", "80 point",
                                                         "75 point", "70 point", "65 point", "60 point", "55 point",
                                                         "50 point", "45 point", "40 point", "35 point", "30 point",
                                                         "25 point", "20 point", "15 point", "10 point", "5 point",
                                                         "0 point", 'Not Applicable'])
        df_full['srch_query_affinity_score'] = df_full['srch_query_affinity_score'].astype('category', copy=False)

        df_full['prop_location_score2'] = df_full['prop_location_score2'].fillna(0)
        df['prop_location_score2'] = pd.cut(df['prop_location_score2'],
                                            bins=[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                                  0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                                                  0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
                                            include_lowest=True)
        df_full['prop_location_score2'] = df_full['prop_location_score2'].astype('category', copy=False)

        #End of questionable changes

        print(df_full.dtypes)
        print(df_full['price_usd'].mean())
        z1 = df_full.loc[df_full["price_usd"] > 5000]
        print(f"Above 5.000: {len(z1)}")
        z1 = df_full.loc[df_full["price_usd"] == 5000]
        print(f"Equal 5.000: {len(z1)}")
        plt.hist(df_full.visitor_hist_adr_usd, 150, density=True, align='mid')
        plt.show()

        """ prop_starrating NA: → transform to vector → categories, keep zeros 
            prop_review_score: NA: same as above
            property_location_score1 → NA: keep zeros
            property_location_score2 → also keep zeros,  (important predictor)  
            price_usd → all to 5000
            origin_dest_dist → NA: take mean
            
            Maybe:
            visitor_hist_starrating: replace missing by mean
            visitor_hist_star_rating → NA: model? mean?  
            Srch_booking_window → check for extreme outliers, correlate with price_usd? 
            srch_query _affinity score: categorize it???
            Comp_rate → combine into better priced, worse priced → reduce features 
        """


    @staticmethod
    def __split_dataset__(dataset, train_percentage, feature_headers, target_header):
        """
        Split the dataset with train_percentage
        :param dataset:
        :param train_percentage:
        :param feature_headers:
        :param target_header:
        :return: train_x, test_x, train_y, test_y
        """

        # Split dataset into train and test dataset
        train_x, test_x, train_y, test_y = train_test_split(dataset[0:2020887][feature_headers],
                                                            dataset[0:2020887][target_header], train_size=train_percentage)
        sk.model_selection.train_test_split()
        return train_x, test_x, train_y, test_y
