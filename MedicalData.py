import csv
from sklearn.naive_bayes import GaussianNB
import math

# Naiven Baesov klasifikator

if __name__ == '__main__':
    with open('medical_data.csv') as csv_folder:
        csv_reader = csv.reader(csv_folder, delimiter=',')
        dataset = list(csv_reader)[1:]
        dataset = [[int(dataset[i][j]) for j in range(0, len(dataset[i]))] for i in range(0, len(dataset))]
        train_set = dataset[0:math.ceil(0.8 * len(dataset))]
        test_set = dataset[math.ceil(0.8 * len(dataset)):]
        train_x = [train_set[i][:-1] for i in range(0, len(train_set))]
        train_y = [train_set[i][-1] for i in range(0, len(train_set))]
        klasifikator = GaussianNB()
        klasifikator.fit(train_x, train_y)
        tocni = 0
        for row in test_set:
            test_x = [row[:-1]]
            prediction = klasifikator.predict(test_x)
            if prediction == row[-1]:
                tocni += 1
        tocnost = tocni / len(test_set)
        print(f'Tocnosta iznesuva: {tocnost}')

        # entry1 = int(input())
        # entry2 = int(input())
        # print(klasifikator.predict([[entry1,entry2]]))

        # line = input()
        # line = line.split(' ')
        # line[0] = int(line[0])
        # line[1] = int(line[1])
        # print(klasifikator.predict([line]))
