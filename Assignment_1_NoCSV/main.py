from csv_reader.reader import reader
from Task_1.subtask_a import subtask_a
from Task_2.t2s1 import t2s1

if __name__ == '__main__':
    file_csv = reader.get_csv('dataset/ODI-2021.csv')
    s1a = subtask_a(file_csv)
    s1a.data_explore()
    # t2 = t2s1("dataset/Titanic/gender_submission.csv", "dataset/Titanic/train.csv", "dataset/Titanic/test.csv")
    # t2.titanic()