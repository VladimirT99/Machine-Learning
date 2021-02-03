import csv
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
import math

# Drva na odluki

if __name__ == '__main__':
    with open('car.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        dataset = list(csv_reader)[1:]
        encoder = OrdinalEncoder()
        encoder.fit([row[:-1] for row in dataset])
        train_set = dataset[:math.ceil(0.7 * len(dataset))]
        test_set = dataset[math.ceil(0.7 * len(dataset)):]

        train_x = [dataset[i][:-1] for i in range(0, len(train_set))]
        train_x = encoder.transform(train_x)
        train_y = [dataset[i][-1] for i in range(0, len(train_set))]
        test_x = [dataset[i][:-1] for i in range(0, len(test_set))]
        test_x = encoder.transform(test_x)
        test_y = [i[-1] for i in test_set]

        klasifikator = DecisionTreeClassifier(criterion='entropy')
        klasifikator.fit(train_x, train_y)

        print(f'Depth: {klasifikator.get_depth()}')
        print(f'Leaves: {klasifikator.get_n_leaves()}')

        tocni = 0
        for row in test_set:
            test_x = [row[:-1]]
            test_x = encoder.transform(test_x)
            test_y = [row[-1]]
            if klasifikator.predict(test_x) == test_y:
                tocni += 1

        print(f'Tocnosta na klasifikatorot iznesuva: {tocni / len(test_set)}')
        print(f'Feature importances: {klasifikator.feature_importances_}')
        features = list(klasifikator.feature_importances_)
        print(f'Najbitna karakteristika: {features.index(max(features))}')
        print(f'Najmalku bitna karakteristika: {features.index(min(features))}')

        dataset_min = []
        for row in dataset:
            dataset_min.append([row[i] for i in range(0, len(row)) if i != features.index(min(features))])

        train_set_min = dataset_min[0:math.ceil(0.7 * len(dataset_min))]
        test_set_min = dataset_min[math.ceil(0.7 * len(dataset_min)):]
        encoder.fit([i[:-1] for i in dataset_min])

        train_x_min = [dataset_min[i][:-1] for i in range(0, len(train_set_min))]
        train_x_min = encoder.transform(train_x_min)
        train_y_min = [dataset_min[i][-1] for i in range(0, len(train_set_min))]

        klasifikator.fit(train_x_min, train_y_min)
        tocni = 0
        for row in test_set_min:
            test_x_min = [row[:-1]]
            test_x_min = encoder.transform(test_x_min)
            test_y_min = [row[-1]]
            if klasifikator.predict(test_x_min) == test_y_min:
                tocni += 1
        print(f'Tocnosta na minimiziraniot dataset e: {tocni / len(test_set_min)}')