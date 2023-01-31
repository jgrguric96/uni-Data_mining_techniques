from read_data import readData

if __name__ == '__main__':
    model = readData("Data/training_set_VU_DM.csv", "Data/test_set_VU_DM.csv")
    model.basic_data_info()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
