
import numpy as np
import pandas
from matplotlib import pyplot as plt
import seaborn
import math

data = pandas.read_csv("train.csv", na_values="NaN")
real_features = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", "Employment_Info_6",
                 "Insurance_History_5", "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5"]
discrete_features = ["Medical_History_1", "Medical_History_10", "Medical_History_15", "Medical_History_24", "Medical_History_32"]
cat_features = data.columns.drop(real_features).drop(discrete_features).drop(["Id", "Response"]).tolist()


print("Half of data is nan ")


def nan_half_data_search():
    for feature in list(data[real_features].columns.values):
        nan_counter = 0
        for val in data[real_features][feature].values:
            if math.isnan(val):
                nan_counter += 1
                if nan_counter > (len(data[real_features][feature]) / 2):
                    print(feature)
                    break


nan_half_data_search()
print("complete")
