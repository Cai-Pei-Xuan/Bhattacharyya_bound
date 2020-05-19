# -*-coding:UTF-8 -*-
# Bhattacharyya bound
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from itertools import combinations

# 載入資料，第一筆是label，其他是feature。
def loadData(file_name, label_list, feature_list):
    fp = open(file_name, "r", encoding="utf-8")
    line = fp.readline()
    while line:
        feature = []
        line_list = line.replace("\n", "").split(",")
        for index, element in enumerate(line_list):
            if index == 0:
                label_list.append(element)
            else:
                feature.append(float(element))
        feature_list.append(feature)
        line = fp.readline()
    fp.close()


# 將train_data中各個相同label的資料放在一起。
def getEachLabelDict(EachLabelDict, label_list, feature_list):
    for i, element in enumerate(label_list):
        if element not in EachLabelDict:
            EachLabelDict[element] = {}
            EachLabelDict[element]["count"] = 1
            EachLabelDict[element]["feature_list"] = []
            EachLabelDict[element]["feature_list"].append(feature_list[i])
        else:
            EachLabelDict[element]["count"] += 1
            EachLabelDict[element]["feature_list"].append(feature_list[i])


# 計算每個label的參數(count, mean, covariance matrix, determinant of covariance matrix)，並將其存成dict，避免日後重複計算。
def calculateEachLabelParameters(EachLabelDict, EachLabelParametersDict):
    for key in EachLabelDict.keys():
        EachLabelParametersDict[key] = {}
        # calculate data number
        EachLabelParametersDict[key]["count"] = EachLabelDict[key]["count"]
        # calculate mean
        EachLabelParametersDict[key]["mean"] = np.mean(
            EachLabelDict[key]["feature_list"], axis=0
        )
        # calculate covariance matrix
        EachLabelParametersDict[key]["cov"] = np.cov(
            np.transpose(EachLabelDict[key]["feature_list"])
        )
        # calculate determinant of covariance matrix
        EachLabelParametersDict[key]["det_of_cov"] = det(
            EachLabelParametersDict[key]["cov"]
        )


# 計算Bhattacharyya bound
def calculateBhattacharyyaBound(class1, class2):
    # 計算μ(s)，且s=1/2。μ(s) is called the Chernoff distance。
    difference_mean = class2["mean"] - class1["mean"]
    average_cov = (class1["cov"] + class2["cov"]) / 2
    average_inv_of_cov = inv(average_cov)
    sqrt_det_of_cov = np.sqrt(class1["det_of_cov"] * class2["det_of_cov"])
    part1 = (
        np.transpose(difference_mean).dot(average_inv_of_cov).dot(difference_mean) / 8
    )
    part2 = np.log(det(average_cov) / sqrt_det_of_cov) / 2
    ChernoffDistance = part1 + part2

    # 計算probability1和probability2
    probability1 = class1["count"] / (class1["count"] + class2["count"])
    probability2 = class2["count"] / (class1["count"] + class2["count"])

    return np.sqrt(probability1 * probability2) * np.exp(-ChernoffDistance)


# 計算所有class的組合的Bhattacharyya bound
def calculateAllClassCombinationsBhattacharyyaBound(EachLabelParametersDict):
    all_class_list = sorted(EachLabelParametersDict.keys())
    combinations_list = list(combinations(all_class_list, 2))
    for combination in combinations_list:
        # 計算Bhattacharyya bound
        BhattacharyyaBound = calculateBhattacharyyaBound(
            EachLabelParametersDict[combination[0]],
            EachLabelParametersDict[combination[1]],
        )

        print(
            combination[0]
            + "和"
            + combination[1]
            + "的Bhattacharyya bound:"
            + str(BhattacharyyaBound)
        )


if __name__ == "__main__":
    file_name = "wine.data"
    label_list = []
    feature_list = []
    EachLabelDict = {}
    EachLabelParametersDict = {}

    # 載入資料
    loadData(file_name, label_list, feature_list)
    # 將train_data中各個相同label的資料放在一起
    getEachLabelDict(EachLabelDict, np.array(label_list), np.array(feature_list))
    # 計算每個label的參數
    calculateEachLabelParameters(EachLabelDict, EachLabelParametersDict)
    # 計算所有class的組合的Bhattacharyya bound
    calculateAllClassCombinationsBhattacharyyaBound(EachLabelParametersDict)
