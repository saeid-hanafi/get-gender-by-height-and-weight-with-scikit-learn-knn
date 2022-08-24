# Libraries
# import scikit learn for KNN Classification for AI
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# import numpy for convert python array to vertical matrix
import numpy as np
# import xlrd for read information from Excel files
import xlrd


# Functions
# function for read information from Excel dataset
def read_excel_file(file_loc):
    file = xlrd.open_workbook(file_loc)
    return file.sheet_by_index(0)


# get dateset info from Excel file and convert to python list
dataset_loc = "Gender_DataSet.xls"
dataset_info = read_excel_file(dataset_loc)
dataset = []

for i in range(0, dataset_info.nrows):
    data_item = [dataset_info.cell_value(i, 0), dataset_info.cell_value(i, 1), dataset_info.cell_value(i, 2)]
    dataset.append(data_item)

# get x train and y train from dataset and convert to vertical matrix them
x_train = []
y_train = []
for j in range(0, len(dataset)):
    x_items = [dataset[j][0], dataset[j][1]]
    x_train.append(x_items)
    y_train.append(dataset[j][2])

x_train_matrix = np.array(x_train).reshape(-1, 2)
y_train_matrix = np.array(y_train).reshape(-1, 1)

# create test array and convert to vertical matrix
test_array = [[180, 95], [160, 70], [190, 83], [174, 81], [172, 70], [195, 110], [162, 78]]
test_matrix = np.array(test_array).reshape(-1, 2)

# get final result by KNN
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train_matrix, y_train_matrix)
predict_list = knn.predict(test_matrix)

print(test_array)
print(predict_list)
