import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def Individual1(data):
    p = (((np.cos(((data["Medical_History_40"] + data["Medical_Keyword_3"]) / 2.0)) + (
    (data["Medical_History_23"] >= (-(data["Medical_History_4"]))).astype(float))) - np.maximum((data["BMI"]), (
    np.tanh((np.tanh(data["SumMedicalKeywords"]) - (data["Product_Info_4"] - data["Medical_Keyword_3"])))))) +
         (np.tanh(((data["Medical_History_20"] + ((data["InsuredInfo_6"] + (((data["Medical_History_15"] - (
         data["InsuredInfo_7"] + data["Insurance_History_2"])) - data["Medical_History_30"]) - data[
                                                                                "Medical_History_5"])) / 2.0)) / 2.0)) - (
          (np.ceil(data["Product_Info_4"]) < data["InsuredInfo_5"]).astype(float))) +
         np.tanh(((((data["Medical_History_11"] + ((
                                                   data["Medical_History_27"] + np.maximum((data["Medical_Keyword_41"]),
                                                                                           (data[
                                                                                                "Family_Hist_4"]))) / 2.0)) / 2.0) + (
                   (((np.sin(np.ceil(data["Product_Info_2_N"])) + data["Insurance_History_5"]) / 2.0) + (((data[
                                                                                                               "Medical_History_13"] -
                                                                                                           data[
                                                                                                               "Medical_History_18"]) +
                                                                                                          data[
                                                                                                              "Family_Hist_2"]) / 2.0)) / 2.0)) / 2.0)) +
         ((data["Medical_History_35"] * data["InsuredInfo_2"]) + (-(((data["Medical_Keyword_38"] >= (-((((data[
                                                                                                              "Ins_Age"] *
                                                                                                          data[
                                                                                                              "Medical_History_31"]) + (
                                                                                                         (data[
                                                                                                              "Medical_History_31"] <= (
                                                                                                          (data[
                                                                                                               "Medical_History_7"] <= np.minimum(
                                                                                                              (data[
                                                                                                                   "Medical_History_28"]),
                                                                                                              (data[
                                                                                                                   "Employment_Info_3"]))).astype(
                                                                                                              float))).astype(
                                                                                                             float))) / 2.0)))).astype(
             float))))) +
         np.minimum((0.318310), (np.maximum(((data["Medical_History_15"] * (
         (0.138462 < (data["Medical_History_23"] + np.minimum((data["Medical_History_15"]), (0.318310)))).astype(
             float)))), (np.maximum((np.minimum((data["Medical_History_4"]), (
         ((data["Medical_History_24"] >= data["Medical_History_23"]).astype(float))))),
                                    (data["Medical_History_15"])))))) +
         ((((data["Medical_History_38"] <= data["Medical_History_24"]).astype(float)) + np.sin(np.minimum((((np.maximum(
             (data["Medical_History_4"]), (data["Medical_History_23"])) + (data["Medical_History_4"] / 2.0)) / 2.0)), ((
                                                                                                                       np.minimum(
                                                                                                                           (
                                                                                                                           data[
                                                                                                                               "Medical_History_11"]),
                                                                                                                           (
                                                                                                                           (
                                                                                                                           data[
                                                                                                                               "Product_Info_2_N"] + np.abs(
                                                                                                                               data[
                                                                                                                                   "Product_Info_2_C"])))) * 2.0))))) / 2.0) +
         (0.094340 * (np.maximum((data["Medical_Keyword_25"]), (np.tanh(data["SumMedicalKeywords"]))) - (((data[
                                                                                                               "InsuredInfo_1"] - (
                                                                                                           (data[
                                                                                                                "Medical_Keyword_22"] +
                                                                                                            data[
                                                                                                                "Medical_Keyword_33"]) / 2.0)) - np.minimum(
             (data["Medical_History_17"]), (data["SumEmploymentInfo"]))) - ((data["Medical_History_1"] + data[
             "Medical_History_33"]) / 2.0)))) +
         np.tanh((data["Medical_History_5"] * (((-(
         (data["Medical_History_3"] + ((data["Medical_History_23"] > np.tanh(data["BMI"])).astype(float))))) + (((data[
                                                                                                                      "Medical_Keyword_9"] - (
                                                                                                                  (data[
                                                                                                                       "Medical_History_4"] >
                                                                                                                   data[
                                                                                                                       "BMI"]).astype(
                                                                                                                      float))) -
                                                                                                                 data[
                                                                                                                     "Medical_Keyword_19"]) -
                                                                                                                data[
                                                                                                                    "Medical_History_6"])) / 2.0))) +
         ((((0.094340 * (
         np.minimum((data["Medical_Keyword_3"]), (((data["BMI"] + data["SumMedicalKeywords"]) / 2.0))) * 2.0)) * np.abs(
             data["BMI"])) + ((np.round((1.0 / (1.0 + np.exp(- np.sin(data["Medical_History_11"]))))) <= np.minimum(
             (data["BMI"]), (data["Ins_Age"]))).astype(float))) / 2.0) +
         (np.minimum((np.tanh(data["Medical_History_31"])), (
         np.maximum(((data["Product_Info_4"] * ((data["BMI"] + ((data["Wt"] + 0.693147) / 2.0)) / 2.0))), (
         np.minimum((data["Medical_History_1"]), (((data["Wt"] + (data["SumNAs"] / 2.0)) / 2.0))))))) / 2.0) +
         (((data["Medical_History_15"] + data["Medical_History_4"]) / 2.0) * (((np.sin((data["Medical_History_4"] * (
         (data["Medical_History_15"] > np.sinh(data["Employment_Info_5"])).astype(float)))) * data[
                                                                                    "Medical_History_15"]) > data[
                                                                                   "Product_Info_3"]).astype(float))) +
         (0.367879 - (1.0 / (1.0 + np.exp(- (
         ((data["Medical_History_40"] + ((data["Ins_Age"] + data["Insurance_History_8"]) / 2.0)) / 2.0) - ((data[
                                                                                                                "Medical_History_4"] <= (
                                                                                                            (data[
                                                                                                                 "Medical_History_23"] <= (
                                                                                                             (data[
                                                                                                                  "BMI"] <= (
                                                                                                              1.0 / (
                                                                                                              1.0 + np.exp(
                                                                                                                  -
                                                                                                                  data[
                                                                                                                      "Medical_History_40"])))).astype(
                                                                                                                 float))).astype(
                                                                                                                float))).astype(
             float))))))) +
         np.minimum((np.minimum((np.minimum((0.058823), ((((data["Employment_Info_1"] + (
         (data["Insurance_History_3"] < np.maximum((data["Ins_Age"]), (data["SumMedicalKeywords"]))).astype(
             float))) / 2.0) / 2.0)))), (np.round(
             np.round(np.maximum((data["Medical_History_23"]), (data["SumMedicalKeywords"]))))))),
                    (np.maximum((data["Medical_History_39"]), (data["SumMedicalKeywords"])))) +
         (np.minimum(
             (np.tanh(((-(data["SumNAs"])) * np.minimum((data["Medical_History_22"]), (data["Insurance_History_2"]))))),
             (data["Medical_Keyword_39"])) * (((((data["Medical_Keyword_2"] - data["Medical_History_19"]) - data[
             "Employment_Info_2"]) - data["Employment_Info_2"]) <= data["Medical_Keyword_37"]).astype(float))) +
         (0.138462 * (((((np.maximum((data["Medical_History_21"]), (data["InsuredInfo_7"])) - data["InsuredInfo_2"]) +
                         data["Medical_History_6"]) / 2.0) + np.round((((data["Product_Info_2_N"] <= data[
             "SumMedicalKeywords"]).astype(float)) * (((data["Family_Hist_3"] >= data["Ins_Age"]).astype(float)) - data[
             "Product_Info_4"])))) / 2.0)) +
         np.sin(np.sinh(np.minimum((data["Medical_History_11"]), (np.sin((data["Medical_History_5"] * ((((np.ceil(
             data["Medical_History_19"]) + ((data["Medical_Keyword_9"] + data["Product_Info_1"]) / 2.0)) / 2.0) > (data[
                                                                                                                       "SumMedicalHistory"] + np.sinh(
             0.720430))).astype(float)))))))) +
         (0.058823 * (((np.maximum((np.maximum((np.maximum((data["Medical_Keyword_45"]), (
         np.maximum((data["Medical_Keyword_34"]), (data["Medical_History_32"]))))), (data["Medical_Keyword_2"]))),
                                   ((data["Medical_Keyword_3"] * data["Medical_History_30"]))) + data[
                            "Product_Info_4"]) + (data["Medical_Keyword_3"] * data["InsuredInfo_6"])) / 2.0)) +
         np.minimum(((data["InsuredInfo_1"] * np.minimum(((data["Medical_History_20"] * (
         0.693147 - (1.0 / (1.0 + np.exp(- np.cos(data["Insurance_History_5"]))))))), ((np.maximum(
             (data["Insurance_History_5"]), ((-(data["Insurance_History_3"])))) - data["Insurance_History_3"]))))),
                    ((data["Insurance_History_5"] * data["Insurance_History_3"]))) +
         np.sin((np.maximum((data["Medical_History_28"]), ((data["Medical_History_24"] * (
         (data["Medical_Keyword_36"] + np.floor((-(data["Medical_History_10"])))) / 2.0)))) * ((((data[
                                                                                                      "Medical_History_13"] >=
                                                                                                  data[
                                                                                                      "Medical_History_10"]).astype(
             float)) <= np.tanh(data["Medical_History_39"])).astype(float)))) +
         np.tanh(((data["BMI"] * (((
                                   ((data["InsuredInfo_7"] >= np.ceil(data["Product_Info_4"])).astype(float)) >= np.sin(
                                       np.maximum((((((data["Ins_Age"] + data["InsuredInfo_2"]) / 2.0) < data[
                                           "Medical_History_4"]).astype(float))),
                                                  (data["Medical_History_15"])))).astype(float)) / 2.0)) / 2.0)) +
         np.minimum((np.sin((data["SumMedicalKeywords"] * np.tanh(((np.ceil(data["Product_Info_4"]) <= (
         (data["Medical_History_30"] + data["Medical_History_18"]) / 2.0)).astype(float)))))), ((data[
                                                                                                     "Product_Info_1"] * (
                                                                                                 (data["BMI"] > ((data[
                                                                                                                      "Medical_History_18"] < np.round(
                                                                                                     data[
                                                                                                         "Medical_Keyword_9"])).astype(
                                                                                                     float))).astype(
                                                                                                     float))))) +
         np.minimum(((0.138462 * ((data["Medical_History_1"] >= data["Medical_History_18"]).astype(float)))), (np.tanh((
                                                                                                                       (
                                                                                                                       (
                                                                                                                       (
                                                                                                                       (
                                                                                                                       (
                                                                                                                       (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_History_1"] >=
                                                                                                                       data[
                                                                                                                           "Medical_History_1"]).astype(
                                                                                                                           float)) <=
                                                                                                                       data[
                                                                                                                           "Family_Hist_3"]).astype(
                                                                                                                           float)) + (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_History_37"] <= np.floor(
                                                                                                                           np.abs(
                                                                                                                               data[
                                                                                                                                   "Product_Info_2_N"]))).astype(
                                                                                                                           float))) / 2.0) +
                                                                                                                       data[
                                                                                                                           "Medical_Keyword_27"]) / 2.0)))) +
         np.minimum((0.094340), ((np.cos(data["Employment_Info_2"]) * np.tanh(((data["Employment_Info_2"] > ((data[
                                                                                                                  "Medical_History_23"] <= (
                                                                                                              np.sin((
                                                                                                                     data[
                                                                                                                         "InsuredInfo_6"] -
                                                                                                                     data[
                                                                                                                         "Medical_History_15"])) -
                                                                                                              data[
                                                                                                                  "Medical_History_39"])).astype(
             float))).astype(float)))))) +
         np.minimum((((data["Medical_History_15"] + ((data["Medical_Keyword_1"] > ((data["Medical_History_15"] + (
         (data["Medical_History_4"] > ((data["Medical_Keyword_10"] > data["Medical_Keyword_23"]).astype(float))).astype(
             float))) / 2.0)).astype(float))) / 2.0)), (((((data["Medical_History_15"] > 8.0).astype(
             float)) >= np.maximum((data["Medical_History_4"]), (data["Medical_History_15"]))).astype(float)))) +
         np.minimum(((np.sin((data["Product_Info_5"] * np.ceil(((data["InsuredInfo_6"] * np.minimum(
             ((data["Medical_History_4"] - data["Medical_Keyword_3"])), (data["Insurance_History_3"]))) + data[
                                                                    "Medical_History_40"])))) / 2.0)),
                    (((np.ceil(data["Medical_History_4"]) < 0.094340).astype(float)))) +
         np.sin(np.minimum((((data["Insurance_History_2"] >= data["Medical_History_20"]).astype(float))), (
         np.maximum((data["Medical_Keyword_3"]), ((np.tanh(data["Medical_Keyword_24"]) * np.minimum(
             (((data["Medical_History_4"] - data["Medical_Keyword_46"]) + (data["Product_Info_3"] + (-(data["BMI"]))))),
             (data["Insurance_History_2"])))))))) +
         np.minimum(((np.cos(np.maximum(((int(0.138462 >= 0.138462))), (np.minimum((data["Ht"]), (2.212120))))) / 2.0)),
                    ((((data["SumMedicalKeywords"] + np.round(data["Ins_Age"])) / 2.0) * ((
                                                                                          data["Medical_History_5"] >= (
                                                                                          (data["Medical_History_17"] +
                                                                                           data[
                                                                                               "Medical_History_20"]) / 2.0)).astype(
                        float))))) +
         (np.minimum((((data["Medical_History_38"] > data["Ins_Age"]).astype(float))), ((0.730769 - (1.0 / (
         1.0 + np.exp(- np.maximum((data["Employment_Info_1"]), (np.maximum((np.maximum((data["Medical_Keyword_47"]), (
         ((data["Medical_History_38"] + data["Medical_History_12"]) / 2.0)))), (np.maximum((data["Ins_Age"]), (
         np.cos(data["InsuredInfo_1"]))))))))))))) / 2.0) +
         (np.ceil(data["Medical_History_4"]) * (0.094340 * np.ceil(((((np.maximum((data["Medical_Keyword_39"]), (
         (data["Product_Info_3"] * np.maximum(((data["Medical_History_23"] * 2.0)), (data["Medical_Keyword_21"]))))) >
                                                                       data["Medical_History_23"]).astype(
             float)) - np.ceil(data["InsuredInfo_2"])) / 2.0)))) +
         (np.round(np.minimum(((-(((data["Insurance_History_5"] > ((data["Insurance_History_7"] < np.tanh(
             ((data["Insurance_History_5"] + (data["Medical_History_20"] * data["Medical_Keyword_31"])) / 2.0))).astype(
             float))).astype(float))))), (((((2.409090 >= data["Ins_Age"]).astype(float)) - data[
             "InsuredInfo_6"]) / 2.0)))) * 2.0) +
         np.maximum((np.sinh(((np.cos((data["Ht"] / 2.0)) < np.minimum((data["Medical_Keyword_14"]),
                                                                       (data["Medical_Keyword_34"]))).astype(float)))),
                    (np.minimum((((data["Medical_Keyword_32"] > np.minimum(((data["SumInsuredInfo"] + 1.584910)), (
                    np.cos(data["Medical_Keyword_39"])))).astype(float))), ((data["Product_Info_2_C"] / 2.0))))) +
         ((np.minimum((data["Medical_History_10"]), (data["Medical_Keyword_14"])) > (-(((data["Medical_History_5"] * (
         data["Medical_History_5"] * ((data["InsuredInfo_6"] >= data["InsuredInfo_3"]).astype(float)))) * ((data[
                                                                                                                "InsuredInfo_3"] >= (
                                                                                                            data[
                                                                                                                "SumEmploymentInfo"] + (
                                                                                                            data[
                                                                                                                "Insurance_History_2"] +
                                                                                                            data[
                                                                                                                "Medical_History_30"]))).astype(
             float)))))).astype(float)) +
         (np.minimum((np.maximum((((data["InsuredInfo_2"] > np.maximum((data["Medical_Keyword_25"]),
                                                                       (data["Medical_History_13"]))).astype(float))),
                                 (data["Medical_Keyword_46"]))), (data["Medical_History_28"])) * ((data[
                                                                                                       "InsuredInfo_2"] >= np.ceil(
             (((data["Medical_History_18"] * (-(data["InsuredInfo_1"]))) + data["Medical_History_22"]) / 2.0))).astype(
             float))) +
         np.tanh((data["Insurance_History_2"] * (
         (data["Medical_History_19"] - (data["Medical_Keyword_44"] - data["Medical_Keyword_31"])) * (((data[
                                                                                                           "Medical_Keyword_17"] *
                                                                                                       data[
                                                                                                           "Medical_History_20"]) > (
                                                                                                      data[
                                                                                                          "Medical_Keyword_17"] * (
                                                                                                      data[
                                                                                                          "Medical_Keyword_30"] * (
                                                                                                      (data[
                                                                                                           "Medical_Keyword_32"] >
                                                                                                       data[
                                                                                                           "Medical_Keyword_39"]).astype(
                                                                                                          float))))).astype(
             float))))) +
         (-(np.minimum((((data["Medical_History_27"] <= np.sin(
             (data["Medical_Keyword_21"] + data["Medical_History_10"]))).astype(float))), (((data[
                                                                                                 "Medical_Keyword_27"] >= (
                                                                                             (-(np.round(np.minimum((
                                                                                                                    np.minimum(
                                                                                                                        (
                                                                                                                        data[
                                                                                                                            "Medical_History_15"]),
                                                                                                                        (
                                                                                                                        data[
                                                                                                                            "Medical_History_3"]))),
                                                                                                                    ((
                                                                                                                     data[
                                                                                                                         "Ins_Age"] / 2.0)))))) * 2.0)).astype(
             float)))))) +
         np.maximum((np.minimum((data["InsuredInfo_5"]),
                                (((data["Medical_Keyword_42"] > data["Medical_History_6"]).astype(float))))), ((np.sin(
             np.maximum((np.maximum((((2.212120 <= (
             data["SumNAs"] - ((data["Medical_Keyword_42"] > data["Employment_Info_2"]).astype(float)))).astype(
                 float))), (np.ceil(data["Medical_Keyword_26"])))), (data["Medical_Keyword_27"]))) / 2.0))) +
         ((((data["Family_Hist_3"] > ((data["Medical_History_33"] + (5.428570 + (
         (data["Wt"] + ((data["Product_Info_3"] <= data["Product_Info_7"]).astype(float))) / 2.0))) / 2.0)).astype(
             float)) + (
           (data["Family_Hist_2"] < (data["Ins_Age"] - (5.200000 + data["Family_Hist_1"]))).astype(float))) / 2.0) +
         (np.maximum((data["Medical_History_4"]), ((-(data["Medical_Keyword_5"])))) * (-(((np.tanh(
             ((data["InsuredInfo_5"] + data["Medical_History_40"]) / 2.0)) > np.cos((((np.minimum((9.869604), (
         data["Medical_Keyword_26"])) >= data["InsuredInfo_6"]).astype(float)) / 2.0))).astype(float))))) +
         np.floor(np.cos(((((data["SumFamilyHistory"] < data["InsuredInfo_6"]).astype(float)) + np.minimum(
             (data["Product_Info_3"]), (((np.floor((data["Medical_Keyword_14"] * (
             -(((data["Product_Info_1"] >= data["Medical_History_14"]).astype(float)))))) / 2.0) - (
                                         data["Product_Info_3"] * data["Medical_History_11"]))))) / 2.0))) +
         ((np.ceil(np.maximum((np.maximum((np.round(data["Insurance_History_9"])), (data["Medical_History_11"]))),
                              (data["Employment_Info_6"]))) == (((((((data["Medical_History_35"] < data[
             "Medical_Keyword_3"]).astype(float)) * data["Medical_History_39"]) + data["Medical_Keyword_3"]) / 2.0) > ((
                                                                                                                       data[
                                                                                                                           "Medical_History_10"] <
                                                                                                                       data[
                                                                                                                           "Medical_History_11"]).astype(
             float))).astype(float))).astype(float)) +
         np.sin((((((((data["Product_Info_7"] + data["Medical_Keyword_31"]) / 2.0) > data["Medical_Keyword_28"]).astype(
             float)) > np.cos(data["Medical_Keyword_40"])).astype(float)) * np.maximum((data["Medical_Keyword_42"]), (
         (np.maximum((data["Medical_Keyword_42"]), (data["Medical_History_40"])) / 2.0))))) +
         ((((data["Medical_History_8"] + data["Medical_History_28"]) < data["Medical_Keyword_15"]).astype(
             float)) * np.sinh((0.058823 - (0.434294 * ((((((data["Medical_History_12"] - data["Medical_History_30"]) -
                                                            data["Medical_Keyword_3"]) >= data[
                                                               "Medical_History_39"]).astype(float)) >= data[
                                                             "Medical_History_40"]).astype(float)))))) +
         ((np.cos(data["Medical_History_19"]) < np.maximum((((((np.minimum(
             (np.minimum((data["Medical_History_21"]), (data["Medical_History_19"]))),
             (data["Medical_History_18"])) > np.cos(
             np.maximum((data["Medical_History_9"]), (data["Medical_History_7"])))).astype(float)) > np.cos(
             data["Insurance_History_2"])).astype(float))), (np.minimum((data["Insurance_History_2"]),
                                                                        (data["Medical_History_9"]))))).astype(float)) +
         np.minimum((((((((data["SumInsuredInfo"] >= np.maximum((data["Insurance_History_7"]), (data["BMI"]))).astype(
             float)) * data["Medical_Keyword_34"]) > 1.570796).astype(float)) / 2.0)), ((np.cos(
             data["Medical_History_29"]) + (((data["Medical_Keyword_32"] < np.floor(data["Employment_Info_4"])).astype(
             float)) * 2.0)))) +
         (((data["Medical_History_29"] > (1.0 / (1.0 + np.exp(- np.sinh(((((((data["Medical_History_29"] > data[
             "Medical_Keyword_41"]).astype(float)) - ((data["Medical_Keyword_29"] + (
         (data["Medical_History_35"] + data["Medical_Keyword_6"]) / 2.0)) / 2.0)) > data["InsuredInfo_4"]).astype(
             float)) - (data["Medical_Keyword_10"] * data["Medical_Keyword_1"]))))))).astype(float)) / 2.0) +
         ((((data["Medical_History_1"] > data["Product_Info_5"]).astype(float)) < np.minimum(
             (np.maximum((data["Medical_History_40"]), (data["BMI"]))), ((np.maximum((data["Medical_Keyword_21"]),
                                                                                     (data["Medical_Keyword_33"])) * (((
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Product_Info_7"] > np.maximum(
                                                                                                                           (
                                                                                                                           data[
                                                                                                                               "Medical_Keyword_31"]),
                                                                                                                           (
                                                                                                                           data[
                                                                                                                               "Medical_Keyword_31"]))).astype(
                                                                                                                           float)) >
                                                                                                                       data[
                                                                                                                           "Medical_History_40"]).astype(
                 float)))))).astype(float)) +
         np.minimum((((data["Medical_History_33"] * np.floor(np.abs(((data["Ins_Age"] + np.minimum((((np.minimum(
             (data["Medical_Keyword_2"]), (data["Insurance_History_9"])) > data["Medical_Keyword_24"]).astype(float))),
                                                                                                   (data[
                                                                                                        "Medical_Keyword_41"]))) / 2.0)))) / 2.0)),
                    ((data["Medical_Keyword_28"] * (
                    (data["Medical_History_21"] > data["Medical_Keyword_38"]).astype(float))))) +
         np.sin(((data["Employment_Info_5"] * (((((data["Medical_History_19"] > (
         data["Medical_Keyword_6"] * data["Ins_Age"])).astype(float)) + data["Medical_History_32"]) > (
                                                (data["Employment_Info_5"] > data["Medical_History_19"]).astype(
                                                    float))).astype(float))) * (
                 (data["Medical_Keyword_6"] < data["Product_Info_4"]).astype(float)))) +
         np.maximum((np.sin(np.minimum((data["InsuredInfo_1"]),
                                       (((data["Medical_Keyword_8"] > data["Medical_History_40"]).astype(float)))))), ((
                                                                                                                       data[
                                                                                                                           "InsuredInfo_1"] * np.tanh(
                                                                                                                           np.ceil(
                                                                                                                               (
                                                                                                                               -(
                                                                                                                               (
                                                                                                                               data[
                                                                                                                                   "Medical_History_6"] * (
                                                                                                                               (
                                                                                                                               data[
                                                                                                                                   "Medical_Keyword_48"] <= np.maximum(
                                                                                                                                   (
                                                                                                                                   data[
                                                                                                                                       "Medical_History_5"]),
                                                                                                                                   (
                                                                                                                                   data[
                                                                                                                                       "Product_Info_1"]))).astype(
                                                                                                                                   float)))))))))) +
         np.ceil(np.sin(np.round(np.maximum((np.maximum(((data["Medical_Keyword_28"] * data["Medical_Keyword_20"])), (
         ((data["Family_Hist_4"] * data["Medical_Keyword_43"]) - data["Medical_Keyword_46"])))), (np.sinh(
             np.minimum((data["Medical_Keyword_19"]),
                        (((data["Medical_History_5"] >= np.cos(data["Medical_Keyword_26"])).astype(float)))))))))) +
         np.tanh((-((data["Medical_Keyword_4"] * ((data["Medical_Keyword_23"] <= (data["InsuredInfo_7"] - (((data[
                                                                                                                 "Medical_Keyword_14"] *
                                                                                                             data[
                                                                                                                 "Medical_Keyword_15"]) < (
                                                                                                            ((data[
                                                                                                                  "Medical_History_37"] -
                                                                                                              data[
                                                                                                                  "Medical_Keyword_23"]) < (
                                                                                                             (data[
                                                                                                                  "Medical_History_22"] >
                                                                                                              data[
                                                                                                                  "Medical_Keyword_12"]).astype(
                                                                                                                 float))).astype(
                                                                                                                float))).astype(
             float)))).astype(float)))))) +
         np.minimum((0.138462), ((((np.abs(data["Ins_Age"]) / 2.0) >= ((((data["Ins_Age"] < np.round((((data[
                                                                                                            "Medical_History_12"] <= 0.730769).astype(
             float)) + ((data["InsuredInfo_3"] >= data["Medical_Keyword_10"]).astype(float))))).astype(
             float)) != np.abs(np.floor(data["Product_Info_2_C"]))).astype(float))).astype(float)))) +
         np.sin(np.maximum((data["Medical_Keyword_43"]), ((0.094340 * np.minimum(((((data["Medical_Keyword_15"] < data[
             "Medical_History_32"]).astype(float)) + (data["Medical_History_26"] * (
         (data["Insurance_History_8"] <= data["Product_Info_2_C"]).astype(float))))), ((data["BMI"] * np.tanh(
             data["Product_Info_4"])))))))) +
         (((((data["Medical_Keyword_41"] >= data["SumMedicalKeywords"]).astype(float)) * (-(((((data[
                                                                                                    "Medical_History_12"] > (
                                                                                                data[
                                                                                                    "Medical_Keyword_38"] *
                                                                                                data[
                                                                                                    "Medical_History_1"])).astype(
             float)) > np.sin(
             np.maximum((np.cos(np.minimum((data["Medical_History_13"]), (data["Medical_History_39"])))),
                        (data["Medical_History_1"])))).astype(float))))) / 2.0) / 2.0) +
         np.minimum((np.maximum((np.cos(data["Product_Info_2_N"])), (data["Medical_History_37"]))), ((0.058823 * ((((
                                                                                                                    data[
                                                                                                                        "Product_Info_2_N"] >= np.maximum(
                                                                                                                        (
                                                                                                                        0.058823),
                                                                                                                        (
                                                                                                                        data[
                                                                                                                            "Medical_History_3"]))).astype(
             float)) >= np.maximum((np.maximum((data["Product_Info_6"]), (data["SumInsuranceHistory"]))),
                                   ((data["Product_Info_2_N"] * data["Medical_History_4"])))).astype(float))))) +
         (np.ceil(np.sin((((np.minimum((data["Medical_Keyword_38"]), (data["Medical_Keyword_13"])) > (
         1.0 / (1.0 + np.exp(- data["Medical_History_7"])))).astype(float)) + np.minimum((data["Medical_History_28"]), (
         ((data["Medical_Keyword_38"] + data["Medical_Keyword_5"]) * (
         1.0 / (1.0 + np.exp(- (data["Medical_History_7"] + data["Product_Info_2_N"])))))))))) * 2.0) +
         (data["BMI"] * (
         np.ceil((data["Ins_Age"] - ((data["Insurance_History_3"] > data["InsuredInfo_1"]).astype(float)))) * np.abs(((
                                                                                                                      0.058823 + np.minimum(
                                                                                                                          (
                                                                                                                          0.138462),
                                                                                                                          (
                                                                                                                          (
                                                                                                                          (
                                                                                                                          data[
                                                                                                                              "Ins_Age"] >= (
                                                                                                                          (
                                                                                                                          data[
                                                                                                                              "InsuredInfo_1"] <=
                                                                                                                          data[
                                                                                                                              "Ins_Age"]).astype(
                                                                                                                              float))).astype(
                                                                                                                              float))))) / 2.0)))) +
         ((np.tanh(data["Medical_Keyword_40"]) * ((data["Medical_Keyword_38"] * np.maximum((data["Product_Info_4"]), (((
                                                                                                                       np.maximum(
                                                                                                                           (
                                                                                                                           data[
                                                                                                                               "Product_Info_4"]),
                                                                                                                           (
                                                                                                                           (
                                                                                                                           (
                                                                                                                           data[
                                                                                                                               "Medical_History_29"] >=
                                                                                                                           data[
                                                                                                                               "SumMedicalKeywords"]).astype(
                                                                                                                               float)))) >=
                                                                                                                       data[
                                                                                                                           "SumMedicalKeywords"]).astype(
             float))))) * np.abs(((data["Medical_History_3"] + data["Medical_History_12"]) / 2.0)))) / 2.0) +
         np.minimum(((-(np.minimum((data["Medical_Keyword_33"]), (
         ((np.sin(data["Medical_Keyword_33"]) >= np.cos(data["Medical_History_24"])).astype(float))))))), (np.floor(
             np.cos((np.sin(data["Medical_Keyword_46"]) - (
             (data["Medical_Keyword_23"] > (data["Medical_Keyword_43"] * data["Medical_Keyword_47"])).astype(
                 float))))))) +
         (0.138462 * np.sin((((np.maximum((np.minimum((data["BMI"]), (data["BMI"]))),
                                          (np.round(data["Medical_Keyword_43"]))) == (
                               (data["Product_Info_4"] > data["Medical_History_40"]).astype(float))).astype(float)) *
                             data["SumMedicalKeywords"]))) +
         np.round((np.minimum(((data["Ht"] + data["Product_Info_5"])), (((data["Employment_Info_5"] + ((data[
                                                                                                            "Medical_Keyword_6"] + (
                                                                                                        data[
                                                                                                            "Medical_History_7"] + (
                                                                                                        (data[
                                                                                                             "Medical_Keyword_13"] +
                                                                                                         data[
                                                                                                             "Employment_Info_5"]) / 2.0))) / 2.0)) / 2.0))) * (
                   (data["Product_Info_5"] > (data["Wt"] + 2.212120)).astype(float)))) +
         ((((data["Medical_History_9"] > data["Medical_History_41"]).astype(float)) <= np.minimum(
             (data["InsuredInfo_5"]), (((np.maximum((data["InsuredInfo_1"]), (((np.maximum((data["Medical_Keyword_12"]),
                                                                                           ((data["Medical_History_9"] *
                                                                                             data[
                                                                                                 "Medical_History_37"]))) + (
                                                                                (data["BMI"] + data[
                                                                                    "Medical_Keyword_12"]) / 2.0)) / 2.0))) +
                                         data["Medical_Keyword_1"]) / 2.0)))).astype(float)) +
         np.tanh(((data["Medical_Keyword_12"] - 0.367879) * ((data["Medical_History_39"] >= ((1.0 > np.minimum(
             (data["Ins_Age"]), ((data["Medical_History_18"] - np.minimum((data["Medical_Keyword_14"]), ((np.maximum(
                 (data["Medical_Keyword_14"]), (data["Medical_History_39"])) - data["Medical_History_32"]))))))).astype(
             float))).astype(float)))) +
         ((-(((((data["Insurance_History_7"] >= ((-(data["Insurance_History_4"])) / 2.0)).astype(float)) == ((data[
                                                                                                                  "Insurance_History_9"] < np.minimum(
             (data["Medical_Keyword_41"]),
             (((np.tanh(data["Insurance_History_9"]) + (data["Medical_Keyword_13"] / 2.0)) / 2.0)))).astype(
             float))).astype(float)))) * 2.0))
    return p + 4.5


def Individual2(data):
    p = ((((((data["Product_Info_4"] + data["Medical_History_4"]) > data["Medical_Keyword_3"]).astype(float)) - ((((
                                                                                                                   data[
                                                                                                                       "SumMedicalKeywords"] + np.maximum(
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "BMI"]),
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_Keyword_3"]))) / 2.0) +
                                                                                                                  data[
                                                                                                                      "BMI"]) / 2.0)) + np.sin(
        ((data["Medical_History_23"] > ((data["Medical_Keyword_3"] + data["BMI"]) / 2.0)).astype(float)))) +
         ((((((((data["Medical_History_15"] - data["InsuredInfo_5"]) - data["Medical_History_5"]) + (
         (data["Family_Hist_4"] - data["Insurance_History_2"]) - data["InsuredInfo_7"])) / 2.0) + (
             (data["Medical_History_40"] + (-(data["Medical_History_30"]))) / 2.0)) / 2.0) + (
           (data["Medical_History_15"] + data["Medical_History_4"]) / 2.0)) / 2.0) +
         np.tanh(np.minimum(((data["Medical_History_20"] + data["Medical_History_11"])), (((np.ceil(
             np.minimum((data["Product_Info_4"]), (
             ((data["Medical_History_7"] + (data["Medical_History_27"] + (-(data["Medical_Keyword_38"])))) / 2.0)))) + (
                                                                                            (data["BMI"] + np.maximum(
                                                                                                (data["InsuredInfo_6"]),
                                                                                                (data[
                                                                                                     "Product_Info_4"]))) / 2.0)) / 2.0)))) +
         ((((data["Medical_History_24"] >= data["Medical_Keyword_41"]).astype(float)) + (((((np.minimum(
             ((data["Employment_Info_2"] * np.round(data["Medical_History_28"]))), (data["Product_Info_2_N"])) + data[
                                                                                                 "Medical_History_13"]) + (
                                                                                            (
                                                                                            data["Medical_Keyword_41"] +
                                                                                            data[
                                                                                                "Family_Hist_2"]) / 2.0)) / 2.0) + (
                                                                                          (data["Insurance_History_5"] +
                                                                                           data[
                                                                                               "Medical_Keyword_25"]) / 2.0)) / 2.0)) / 2.0) +
         (0.138462 * (data["SumMedicalKeywords"] + ((((((data["Medical_History_31"] - data["Medical_History_35"]) - (
         data["Medical_History_18"] - data["Medical_Keyword_33"])) - data["InsuredInfo_1"]) + np.maximum(
             (data["Medical_History_21"]), (data["InsuredInfo_6"]))) / 2.0) - data["Ins_Age"]))) +
         np.minimum((((np.sin(np.maximum((data["InsuredInfo_2"]), (data["Medical_History_15"]))) + ((np.tanh(((((data[
                                                                                                                     "Medical_History_1"] +
                                                                                                                 data[
                                                                                                                     "Medical_History_3"]) / 2.0) +
                                                                                                               data[
                                                                                                                   "Medical_History_33"]) / 2.0)) / 2.0) - (
                                                                                                    (data[
                                                                                                         "InsuredInfo_2"] >=
                                                                                                     data[
                                                                                                         "Medical_History_17"]).astype(
                                                                                                        float)))) / 2.0)),
                    (np.cos((-(data["Ins_Age"]))))) +
         np.minimum((np.sin(np.maximum((np.abs(data["Medical_History_11"])), ((data["Medical_Keyword_9"] * 2.0))))), ((
                                                                                                                      np.sin(
                                                                                                                          np.abs(
                                                                                                                              np.maximum(
                                                                                                                                  (
                                                                                                                                  np.maximum(
                                                                                                                                      (
                                                                                                                                      data[
                                                                                                                                          "Medical_Keyword_15"]),
                                                                                                                                      (
                                                                                                                                      data[
                                                                                                                                          "SumNAs"]))),
                                                                                                                                  (
                                                                                                                                  data[
                                                                                                                                      "Medical_History_11"])))) + (
                                                                                                                      (
                                                                                                                      np.abs(
                                                                                                                          data[
                                                                                                                              "Family_Hist_4"]) +
                                                                                                                      data[
                                                                                                                          "SumNAs"]) / 2.0)))) +
         np.tanh(np.sin(((data["Medical_History_15"] <= ((np.sin(((((((np.sin(
             (data["Product_Info_2_N"] * 2.0)) >= 0.840000).astype(float)) >= (data["Medical_History_4"] / 2.0)).astype(
             float)) > np.abs(data["Product_Info_4"])).astype(float))) / 2.0) - 0.840000)).astype(float)))) +
         (((data["Medical_Keyword_47"] < ((data["Medical_History_15"] + data["Medical_History_22"]) / 2.0)).astype(
             float)) * np.minimum((((data["Medical_History_15"] == 0.094340).astype(float))), ((((data[
                                                                                                      "Medical_History_15"] + (
                                                                                                  (data["BMI"] + data[
                                                                                                      "Medical_History_15"]) +
                                                                                                  data[
                                                                                                      "Medical_History_15"])) / 2.0) +
                                                                                                data[
                                                                                                    "Medical_History_15"])))) +
         ((((((data["Medical_History_1"] >= data["Medical_History_30"]).astype(float)) / 2.0) + (-(((data[
                                                                                                         "Medical_History_6"] <= (
                                                                                                     (np.cos(data[
                                                                                                                 "Wt"]) <= (
                                                                                                      (((data[
                                                                                                             "Medical_History_18"] <= np.ceil(
                                                                                                          data[
                                                                                                              "Product_Info_4"])).astype(
                                                                                                          float)) <= ((
                                                                                                                      data[
                                                                                                                          "Medical_History_6"] <=
                                                                                                                      data[
                                                                                                                          "Medical_History_30"]).astype(
                                                                                                          float))).astype(
                                                                                                          float))).astype(
                                                                                                         float))).astype(
             float))))) / 2.0) / 2.0) +
         (((((1.0 / (1.0 + np.exp(- np.floor(data["Medical_History_15"])))) > (np.abs(data["Medical_History_15"]) + (
         (data["Product_Info_4"] > ((data["Family_Hist_5"] != data["BMI"]).astype(float))).astype(float)))).astype(
             float)) + ((np.minimum((data["Ins_Age"]), (data["BMI"])) > (
         (data["BMI"] != data["Family_Hist_2"]).astype(float))).astype(float))) / 2.0) +
         np.minimum((0.585714), (np.maximum((np.minimum((data["InsuredInfo_6"]), (data["Medical_Keyword_3"]))), ((
                                                                                                                 np.cos(
                                                                                                                     data[
                                                                                                                         "Product_Info_2_C"]) * (
                                                                                                                 (((
                                                                                                                   0.318310 <= (
                                                                                                                   (
                                                                                                                   data[
                                                                                                                       "Medical_Keyword_41"] + np.abs(
                                                                                                                       data[
                                                                                                                           "Product_Info_2_C"])) / 2.0)).astype(
                                                                                                                     float)) + (
                                                                                                                  (data[
                                                                                                                       "Medical_History_10"] +
                                                                                                                   data[
                                                                                                                       "Product_Info_2_N"]) / 2.0)) / 2.0)))))) +
         np.minimum((((np.floor(((data["Family_Hist_3"] + (
         (np.floor(((data["BMI"] + data["Medical_History_19"]) / 2.0)) <= data["Medical_History_39"]).astype(
             float))) / 2.0)) / 2.0) / 2.0)), (np.cos((data["Product_Info_4"] - (
         1.0 / (1.0 + np.exp(- ((data["BMI"] <= np.cos(data["Family_Hist_3"])).astype(float))))))))) +
         np.minimum(((data["SumMedicalKeywords"] * np.sinh(
             ((data["Medical_History_18"] > data["Medical_History_39"]).astype(float))))), (
                    np.minimum((((data["Medical_History_5"] > data["Medical_History_20"]).astype(float))), ((data[
                                                                                                                 "SumMedicalKeywords"] * np.sinh(
                        ((np.sinh(((data["Medical_History_5"] > data["Medical_History_17"]).astype(float))) > data[
                            "Medical_History_20"]).astype(float)))))))) +
         np.maximum(((0.058823 * data["Family_Hist_2"])), (np.tanh(np.minimum((0.301030), (np.maximum(((((((data[
                                                                                                                "Medical_Keyword_34"] +
                                                                                                            data[
                                                                                                                "InsuredInfo_7"]) / 2.0) + (
                                                                                                          (data[
                                                                                                               "Medical_Keyword_19"] > np.cos(
                                                                                                              np.minimum(
                                                                                                                  (data[
                                                                                                                       "SumInsuredInfo"]),
                                                                                                                  (data[
                                                                                                                       "InsuredInfo_7"])))).astype(
                                                                                                              float))) / 2.0) / 2.0)),
                                                                                                      (data[
                                                                                                           "Medical_Keyword_45"]))))))) +
         ((data["Ins_Age"] * np.sin((np.sin(np.minimum((0.367879), (
         np.minimum((np.maximum((data["Wt"]), (data["Medical_Keyword_23"]))),
                    (np.cos(data["Medical_History_18"])))))) / 2.0))) / 2.0) +
         np.minimum((((np.sin(data["Medical_History_22"]) + (((np.sin(np.cos(data["Ins_Age"])) * (
         (data["Family_Hist_5"] + data["Medical_History_29"]) / 2.0)) + data["Medical_Keyword_2"]) / 2.0)) / 2.0)),
                    ((data["Family_Hist_5"] * (0.094340 * np.cos(data["SumNAs"]))))) +
         (((np.minimum((data["Medical_Keyword_6"]), ((((data["Employment_Info_1"] + data["Employment_Info_5"]) + (
         (data["Employment_Info_2"] > data["Insurance_History_8"]).astype(float))) / 2.0))) / 2.0) + (
           np.minimum((0.094340),
                      (((data["InsuredInfo_6"] > data["Insurance_History_8"]).astype(float)))) * 2.0)) / 2.0) +
         (((data["SumMedicalKeywords"] * ((data["InsuredInfo_2"] >= (data["SumMedicalKeywords"] * ((data[
                                                                                                        "Medical_History_30"] >= np.cos(
             (data["InsuredInfo_6"] + (data["Medical_Keyword_15"] + data["Medical_History_23"])))).astype(
             float)))).astype(float))) + np.sin(((-(data["Medical_History_38"])) / 2.0))) / 2.0) +
         np.ceil(np.minimum((np.maximum(
             (np.sin(np.minimum(((data["Medical_Keyword_46"] * 8.0)), (data["InsuredInfo_6"])))),
             (np.sinh(np.minimum((data["Medical_Keyword_34"]), (data["Medical_Keyword_14"])))))),
                            ((data["Insurance_History_5"] + data["Insurance_History_1"])))) +
         np.floor((((1.0 / (1.0 + np.exp(- data["Medical_History_32"]))) + (np.tanh(data["BMI"]) / 2.0)) + np.round((
                                                                                                                    data[
                                                                                                                        "BMI"] * np.tanh(
                                                                                                                        np.minimum(
                                                                                                                            (
                                                                                                                            (
                                                                                                                            data[
                                                                                                                                "Medical_History_11"] *
                                                                                                                            data[
                                                                                                                                "Medical_History_5"])),
                                                                                                                            (
                                                                                                                            np.cos(
                                                                                                                                (
                                                                                                                                1.0 / (
                                                                                                                                1.0 + np.exp(
                                                                                                                                    -
                                                                                                                                    data[
                                                                                                                                        "InsuredInfo_5"]))))))))))) +
         (((np.maximum((data["Wt"]), (data["Medical_History_28"])) - np.cos((data["BMI"] * 2.0))) - (
         (data["Employment_Info_4"] > data["Medical_Keyword_25"]).astype(float))) * np.minimum(
             (data["Medical_History_35"]), (((data["Insurance_History_5"] > data["Wt"]).astype(float))))) +
         np.sin(np.maximum(
             (((data["Medical_Keyword_25"] + np.minimum((data["InsuredInfo_5"]), (data["Medical_Keyword_1"]))) / 2.0)),
             (np.maximum((data["Medical_Keyword_25"]), ((data["Medical_Keyword_11"] * ((data[
                                                                                            "Medical_Keyword_43"] > np.sin(
                 np.round(((np.maximum((data["Medical_Keyword_25"]), (data["Product_Info_5"])) + data[
                     "Medical_History_7"]) / 2.0)))).astype(float)))))))) +
         (((-(np.floor(data["Product_Info_4"]))) * 0.138462) * (np.sin(data["Medical_History_4"]) * np.tanh(
             np.minimum((np.minimum((np.sin(data["Medical_Keyword_42"])), (data["Medical_History_1"]))),
                        (np.round(data["Medical_History_4"])))))) +
         np.minimum((((0.058823 <= np.minimum((data["Medical_History_30"]), (
         ((data["Medical_Keyword_32"] + data["Medical_Keyword_3"]) / 2.0)))).astype(float))), (np.cos(
             np.minimum((np.minimum((data["BMI"]), (data["Medical_History_4"]))),
                        ((-((np.sin(np.sinh(np.sin(np.sinh(data["BMI"])))) * 2.0)))))))) +
         (np.minimum((0.094340), (((((np.maximum((data["Medical_Keyword_37"]), (data["Medical_History_23"])) <= data[
             "BMI"]).astype(float)) + data["Insurance_History_2"]) / 2.0))) * ((data["Medical_Keyword_15"] + (
         ((data["BMI"] * data["Insurance_History_2"]) + data["SumMedicalKeywords"]) / 2.0)) / 2.0)) +
         (np.sin(np.minimum((((((data["Medical_Keyword_12"] + data["Medical_Keyword_15"]) / 2.0) + (
         (data["SumEmploymentInfo"] <= data["Medical_History_13"]).astype(float))) / 2.0)), ((data[
                                                                                                  "Medical_Keyword_15"] * (
                                                                                              -((data[
                                                                                                     "Medical_Keyword_15"] * (
                                                                                                 -(((data[
                                                                                                         "Medical_Keyword_12"] >=
                                                                                                     data[
                                                                                                         "Medical_History_13"]).astype(
                                                                                                     float))))))))))) / 2.0) +
         np.minimum((np.tanh((-(((2.212120 < (data["Ins_Age"] * (
         (data["Medical_History_33"] + ((data["Medical_History_14"] < data["BMI"]).astype(float))) / 2.0))).astype(
             float)))))), (np.tanh((-(((2.212120 < data["Ins_Age"]).astype(float))))))) +
         np.maximum((np.ceil(np.tanh(np.minimum((data["InsuredInfo_5"]), (data["Medical_History_32"]))))), (np.maximum((
                                                                                                                       np.minimum(
                                                                                                                           (
                                                                                                                           data[
                                                                                                                               "Medical_Keyword_19"]),
                                                                                                                           (
                                                                                                                           (
                                                                                                                           (
                                                                                                                           np.cos(
                                                                                                                               data[
                                                                                                                                   "Medical_Keyword_39"]) <=
                                                                                                                           data[
                                                                                                                               "BMI"]).astype(
                                                                                                                               float))))),
                                                                                                                       (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_History_17"] - (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Product_Info_2_C"] <= np.cos(
                                                                                                                           data[
                                                                                                                               "Medical_Keyword_20"])).astype(
                                                                                                                           float))))))) +
         np.sin(np.maximum((data["Medical_Keyword_25"]), (np.sin(np.maximum((data["Medical_Keyword_29"]), (
         np.maximum((data["Medical_Keyword_25"]), ((data["Medical_History_13"] * ((data["SumNAs"] > (
         2.409090 - (((data["Employment_Info_2"] > (2.409090 - data["Product_Info_7"])).astype(float)) * 2.0))).astype(
             float))))))))))) +
         (0.094340 * (-(((((data["Medical_Keyword_37"] >= (-(((((data["Employment_Info_2"] >= ((0.058823 > (
         data["Medical_History_19"] + np.maximum((data["Employment_Info_2"]), (data["Medical_Keyword_48"])))).astype(
             float))).astype(float)) > data["Employment_Info_2"]).astype(float))))).astype(float)) > np.abs(
             data["Employment_Info_5"])).astype(float))))) +
         np.minimum((np.abs((data["Ins_Age"] * data["Medical_Keyword_17"]))), (np.minimum(((0.138462 * ((0.693147 >= (((
                                                                                                                       (
                                                                                                                       (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_History_2"] <
                                                                                                                       data[
                                                                                                                           "Medical_Keyword_11"]).astype(
                                                                                                                           float)) <
                                                                                                                       data[
                                                                                                                           "Medical_History_2"]).astype(
                                                                                                                           float)) * 2.0) -
                                                                                                                      data[
                                                                                                                          "Medical_History_2"])).astype(
             float)))), (np.cos(data["Medical_History_32"]))))) +
         (((data["Product_Info_2_N"] < data["Medical_Keyword_31"]).astype(float)) * np.floor(np.floor(np.cos(((data[
                                                                                                                   "Product_Info_3"] + (
                                                                                                               ((data[
                                                                                                                     "Medical_History_20"] + np.floor(
                                                                                                                   data[
                                                                                                                       "Insurance_History_3"])) / 2.0) * (
                                                                                                               (data[
                                                                                                                    "Medical_Keyword_31"] > (
                                                                                                                (data[
                                                                                                                     "Medical_History_27"] +
                                                                                                                 data[
                                                                                                                     "Medical_Keyword_31"]) / 2.0)).astype(
                                                                                                                   float)))) / 2.0))))) +
         np.minimum((np.maximum((data["Insurance_History_5"]),
                                (((data["Insurance_History_5"] + data["Insurance_History_9"]) / 2.0)))), ((-3.0 * ((
                                                                                                                   0.367879 <= (
                                                                                                                   data[
                                                                                                                       "Insurance_History_5"] - (
                                                                                                                   (
                                                                                                                   np.maximum(
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Insurance_History_7"]),
                                                                                                                       (
                                                                                                                       np.floor(
                                                                                                                           data[
                                                                                                                               "Insurance_History_9"]))) < (
                                                                                                                   (
                                                                                                                   data[
                                                                                                                       "Insurance_History_4"] < 0.0).astype(
                                                                                                                       float))).astype(
                                                                                                                       float)))).astype(
             float))))) +
         (((data["Product_Info_7"] >= (data["Medical_History_38"] * data["SumEmploymentInfo"])).astype(float)) * (
         ((data["Medical_History_4"] >= data["Medical_History_5"]).astype(float)) * (
         (data["Medical_History_13"] + data["Medical_Keyword_40"]) / 2.0))) +
         (((data["BMI"] * (((data["InsuredInfo_7"] + (
         (-(np.maximum((data["Medical_History_23"]), ((data["Medical_History_40"] * data["Medical_History_30"]))))) * (
         (data["Product_Info_4"] + np.minimum((data["Medical_History_20"]), (data["Medical_History_27"]))) / 2.0))) >
                            data["Medical_History_31"]).astype(float))) / 2.0) / 2.0) +
         (-((0.138462 * (((np.cos(data["Medical_History_28"]) * np.sin((((data["Product_Info_2_N"] <= (
         (data["Ht"] == np.abs(0.636620)).astype(float))).astype(float)) * data["Ins_Age"]))) < np.minimum(
             (data["Medical_History_12"]), (data["Product_Info_2_N"]))).astype(float))))) +
         (((((np.minimum((data["Medical_History_28"]),
                         (((data["Product_Info_7"] + data["Medical_Keyword_5"]) / 2.0))) > np.cos(
             np.minimum((data["Medical_Keyword_3"]), (data["Medical_Keyword_40"])))).astype(float)) > ((np.cos(
             data["Medical_Keyword_39"]) + np.maximum((data["Medical_Keyword_10"]),
                                                      (np.cos(data["Medical_History_25"])))) / 2.0)).astype(
             float)) / 2.0) +
         (np.sin((data["Medical_History_5"] * ((data["Medical_History_19"] > (
         data["Product_Info_1"] * np.minimum((data["Medical_Keyword_47"]), (
         ((data["Medical_History_19"] - 0.094340) - np.sinh(data["Product_Info_2_C"])))))).astype(
             float)))) * np.maximum((data["Medical_History_3"]), (data["Medical_History_29"]))) +
         (0.094340 * (np.minimum((((data["Medical_History_41"] >= data["Medical_History_3"]).astype(float))), (np.ceil((
                                                                                                                       (
                                                                                                                       (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_History_4"] * 2.0) * (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_Keyword_3"] > (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_Keyword_23"] >=
                                                                                                                       data[
                                                                                                                           "Medical_Keyword_48"]).astype(
                                                                                                                           float))).astype(
                                                                                                                           float))) +
                                                                                                                       data[
                                                                                                                           "Medical_Keyword_23"]) / 2.0)))) *
                      data["Medical_Keyword_3"])) +
         ((((data["SumEmploymentInfo"] > np.minimum((data["Medical_Keyword_12"]), (data["Family_Hist_5"]))).astype(
             float)) < ((np.minimum((np.minimum(((-(data["Medical_History_7"]))), (data["Family_Hist_5"]))),
                                    (data["Medical_History_21"])) >= (1.0 / (1.0 + np.exp(
             - (data["Medical_Keyword_8"] - (data["Medical_Keyword_12"] - data["Medical_Keyword_8"])))))).astype(
             float))).astype(float)) +
         (np.minimum((((data["InsuredInfo_7"] >= data["Medical_History_32"]).astype(float))),
                     (((data["Medical_History_32"] >= data["Medical_History_20"]).astype(float)))) + (((data[
                                                                                                            "Medical_History_9"] >= np.cos(
             ((np.ceil(data["Medical_History_34"]) == ((data["Medical_History_32"] >= 20.750000).astype(float))).astype(
                 float)))).astype(float)) * 2.0)) +
         ((-(((((data["Ht"] == data["Medical_Keyword_44"]).astype(float)) > np.cos(
             ((data["SumMedicalHistory"] - data["Medical_Keyword_44"]) / 2.0))).astype(float)))) * np.sinh(
             np.abs(((data["Product_Info_6"] <= data["BMI"]).astype(float))))) +
         np.sin((((data["Medical_Keyword_12"] + data["Medical_Keyword_20"]) / 2.0) * (-(((data["Medical_Keyword_26"] > (
         data["Medical_Keyword_10"] * (data["Medical_Keyword_39"] * (
         (np.sin(data["Medical_Keyword_12"]) <= data["Employment_Info_6"]).astype(float))))).astype(float)))))) +
         (data["Medical_Keyword_38"] * (np.minimum((data["Medical_History_30"]), (np.tanh(
             np.minimum((data["Product_Info_3"]), (((np.sinh((data["Medical_Keyword_45"] - (
             data["Medical_Keyword_3"] * data["Medical_Keyword_13"]))) / 2.0) / 2.0)))))) / 2.0)) +
         ((data["SumNAs"] >= (5.200000 - np.maximum(((data["Medical_Keyword_39"] * data["Medical_Keyword_47"])), (
         np.maximum(((np.minimum((data["Medical_History_15"]), (
         np.minimum((5.200000), (((data["SumFamilyHistory"] >= data["Medical_Keyword_3"]).astype(float)))))) * data[
                          "Medical_Keyword_2"])), (np.abs(data["Ht"]))))))).astype(float)) +
         (-(np.minimum((np.minimum((((data["Medical_Keyword_47"] > (
         data["Medical_Keyword_24"] * np.minimum((data["Medical_History_28"]), (
         (((data["Product_Info_1"] > data["Medical_History_13"]).astype(float)) * 2.0))))).astype(float))),
                                   ((1.0 / (1.0 + np.exp(- data["Medical_History_3"])))))),
                       (((data["Insurance_History_8"] >= data["Medical_History_11"]).astype(float)))))) +
         np.floor(np.tanh((data["Medical_Keyword_18"] * (data["Medical_Keyword_17"] * (
         ((data["Medical_History_19"] >= (-(data["Medical_Keyword_45"]))).astype(float)) - (
         data["InsuredInfo_7"] * np.tanh(
             (((data["Medical_History_19"] > (-(data["Medical_Keyword_4"]))).astype(float)) / 2.0)))))))) +
         (-(((((data["Employment_Info_4"] <= data["Medical_Keyword_23"]).astype(float)) > np.floor(((((np.sin(
             data["Medical_History_11"]) <= data["Medical_Keyword_23"]).astype(float)) > ((
                                                                                          data["Medical_Keyword_46"] + (
                                                                                          (data["Medical_History_11"] <
                                                                                           data[
                                                                                               "Medical_Keyword_16"]).astype(
                                                                                              float))) / 2.0)).astype(
             float)))).astype(float)))) +
         np.maximum((((np.minimum((data["Ins_Age"]), ((((np.minimum((data["Medical_History_30"]),
                                                                    (data["Medical_History_10"])) * data[
                                                             "Medical_History_10"]) > np.cos(
             data["Family_Hist_1"])).astype(float)))) == np.ceil(data["Medical_History_6"])).astype(float))), (np.floor(
             np.minimum((data["Employment_Info_3"]),
                        (np.minimum((data["Medical_Keyword_16"]), (data["Employment_Info_6"]))))))) +
         np.sinh(np.abs((((((data["SumMedicalHistory"] * (
         ((data["Employment_Info_1"] / 2.0) <= data["Employment_Info_4"]).astype(float))) > (
                            data["Medical_History_36"] + (9.869604 / 2.0))).astype(float)) > np.cos(
             np.minimum((data["Medical_Keyword_36"]),
                        (((data["Medical_History_28"] + data["SumMedicalHistory"]) / 2.0))))).astype(float)))) +
         np.sin((data["SumMedicalKeywords"] * np.minimum(((1.0 / (1.0 + np.exp(- data["Medical_History_40"])))), (((
                                                                                                                   data[
                                                                                                                       "InsuredInfo_5"] >= (
                                                                                                                   data[
                                                                                                                       "SumMedicalKeywords"] * np.sinh(
                                                                                                                       np.minimum(
                                                                                                                           (
                                                                                                                           (
                                                                                                                           1.0 / (
                                                                                                                           1.0 + np.exp(
                                                                                                                               - np.floor(
                                                                                                                                   data[
                                                                                                                                       "InsuredInfo_5"]))))),
                                                                                                                           (
                                                                                                                           (
                                                                                                                           (
                                                                                                                           data[
                                                                                                                               "Medical_History_19"] >=
                                                                                                                           data[
                                                                                                                               "Medical_History_31"]).astype(
                                                                                                                               float))))))).astype(
             float)))))) +
         ((np.minimum((data["Medical_History_38"]), (data["Medical_Keyword_39"])) / 2.0) * (-(((data[
                                                                                                    "Medical_Keyword_39"] > (
                                                                                                (((np.round(((data[
                                                                                                                  "Medical_History_36"] +
                                                                                                              data[
                                                                                                                  "Product_Info_6"]) / 2.0)) +
                                                                                                   data[
                                                                                                       "SumInsuredInfo"]) / 2.0) / 2.0) / 2.0)).astype(
             float))))) +
         ((((((((data["Medical_History_12"] > data["Medical_History_13"]).astype(float)) > np.floor(
             (data["InsuredInfo_1"] / 2.0))).astype(float)) > ((data["Medical_History_11"] < np.sin(
             np.minimum((data["Medical_Keyword_43"]),
                        (((data["SumEmploymentInfo"] > data["Medical_Keyword_9"]).astype(float)))))).astype(
             float))).astype(float)) < (
           (data["Medical_Keyword_42"] > data["Medical_History_40"]).astype(float))).astype(float)) +
         (-(((((((np.minimum((data["Medical_Keyword_42"]), (data["Medical_Keyword_40"])) > (
         (data["Employment_Info_1"] + (1.0 / (1.0 + np.exp(- data["Employment_Info_4"])))) / 2.0)).astype(float)) > (
                (data["Employment_Info_4"] <= (data["Medical_Keyword_40"] * data["SumInsuredInfo"])).astype(
                    float))).astype(float)) > np.ceil(data["Medical_Keyword_3"])).astype(float)))) +
         (np.tanh(((-(np.maximum((data["Medical_History_4"]), ((data["Medical_Keyword_36"] / 2.0))))) * (
         (data["SumInsuranceHistory"] >= np.ceil(data["Medical_History_40"])).astype(float)))) * ((((data[
                                                                                                         "SumInsuranceHistory"] >= np.ceil(
             data["Medical_History_40"])).astype(float)) >= np.abs(data["Medical_History_28"])).astype(float))) +
         ((data["Medical_History_19"] < np.minimum(((data["Medical_History_19"] * data["Medical_History_15"])), (
         np.minimum((data["Medical_Keyword_3"]), ((-((1.0 / (1.0 + np.exp(- np.round(np.minimum((np.minimum(
             (data["Medical_History_40"]),
             (np.minimum((data["Medical_History_22"]), ((data["Medical_History_35"] * data["Medical_History_19"])))))),
                                                                                                (data[
                                                                                                     "BMI"]))))))))))))).astype(
             float)) +
         ((5.428570 < (data["Medical_History_19"] + (((data["Employment_Info_2"] < np.ceil((
                                                                                           data["Medical_History_9"] * (
                                                                                           (data[
                                                                                                "Insurance_History_2"] >= np.maximum(
                                                                                               (data[
                                                                                                    "Medical_History_5"]),
                                                                                               (data[
                                                                                                    "Medical_Keyword_33"]))).astype(
                                                                                               float))))).astype(
             float)) * ((data["Insurance_History_2"] >= (data["Medical_History_7"] - data["Medical_History_5"])).astype(
             float))))).astype(float)) +
         np.floor((1.0 / (1.0 + np.exp(- (data["Product_Info_5"] + (data["Medical_History_7"] + (
         ((data["Medical_Keyword_14"] + data["Medical_Keyword_32"]) + data["InsuredInfo_7"]) * np.maximum(
             (data["Medical_Keyword_46"]), ((data["Medical_Keyword_39"] * data["SumInsuredInfo"])))))))))) +
         ((1.0 / (1.0 + np.exp(- data["Product_Info_6"]))) * (np.floor(np.sin(np.floor(((np.sin(
             np.maximum(((data["Medical_History_13"] - (-(data["BMI"])))), (np.maximum((data["Employment_Info_2"]), (
             np.maximum((data["Employment_Info_1"]), (data["Medical_History_13"]))))))) / 2.0) / 2.0)))) / 2.0)) +
         (data["Family_Hist_5"] * (((data["Medical_History_28"] * data["Medical_Keyword_13"]) >= (1.0 / (1.0 + np.exp(
             - (np.minimum(((data["Ins_Age"] * data["Medical_History_5"])), (data["InsuredInfo_3"])) + (
             data["Ins_Age"] * data["Medical_History_30"])))))).astype(float))) +
         np.round((np.sin(np.ceil(np.ceil(((((data["Wt"] - data["SumNAs"]) < (
         ((data["Medical_History_40"] < data["Wt"]).astype(float)) * data["Employment_Info_1"])).astype(
             float)) * np.minimum((data["Medical_Keyword_12"]), (data["Medical_Keyword_25"])))))) * 2.0)) +
         (np.minimum((data["Medical_Keyword_44"]), (((np.minimum((data["Medical_Keyword_44"]), (
         ((data["Employment_Info_4"] > data["Wt"]).astype(float)))) > data["Wt"]).astype(float)))) * ((data[
                                                                                                           "Medical_History_12"] >= (
                                                                                                       -(((np.minimum(((
                                                                                                                       data[
                                                                                                                           "Medical_Keyword_44"] / 2.0)),
                                                                                                                      (
                                                                                                                      data[
                                                                                                                          "SumInsuredInfo"])) +
                                                                                                           data[
                                                                                                               "Medical_Keyword_47"]) / 2.0)))).astype(
             float))) +
         (0.138462 * (((data["Ins_Age"] * np.minimum(
             (np.maximum((data["Medical_Keyword_27"]), (data["Medical_Keyword_22"]))),
             (np.cos(data["Medical_History_24"])))) + (
                       np.minimum((np.minimum((data["Medical_History_20"]), (data["Medical_History_13"]))),
                                  (data["Medical_History_23"])) * (data["InsuredInfo_1"] / 2.0))) / 2.0)))
    return p + 4.5


def Individual3(data):
    p = ((((((-(data["BMI"])) + (
    (data["Medical_History_4"] < ((data["BMI"] < data["Medical_History_23"]).astype(float))).astype(float))) + (
            data["Medical_History_4"] - data["SumMedicalKeywords"])) / 2.0) + np.cos(
        ((data["Medical_Keyword_3"] + (data["Medical_History_40"] - data["BMI"])) / 2.0))) +
         ((((data["InsuredInfo_6"] + data["Medical_History_15"]) / 2.0) + ((np.minimum(
             ((np.ceil(np.minimum((data["Product_Info_4"]), ((np.ceil(data["Product_Info_4"]) * 2.0)))) * 2.0)),
             ((-(((data["Insurance_History_2"] + (data["Medical_History_30"] + data["InsuredInfo_7"])) / 2.0))))) +
                                                                            data["Product_Info_4"]) / 2.0)) / 2.0) +
         np.tanh((np.minimum(((np.cos(data["Ins_Age"]) - data["InsuredInfo_5"])),
                             (np.minimum((data["Medical_History_20"]), (data["Medical_History_27"])))) - (
                  data["Medical_History_5"] * (data["Medical_Keyword_22"] + (
                  (data["Family_Hist_4"] + data["SumMedicalKeywords"]) + data["Medical_History_5"]))))) +
         np.tanh((((np.maximum((data["Medical_Keyword_41"]), (data["SumMedicalKeywords"])) + ((data[
                                                                                                   "Insurance_History_5"] + (
                                                                                               -(np.maximum((data[
                                                                                                                 "Medical_History_18"]),
                                                                                                            (np.maximum(
                                                                                                                (
                                                                                                                np.maximum(
                                                                                                                    (
                                                                                                                    data[
                                                                                                                        "Medical_Keyword_3"]),
                                                                                                                    (
                                                                                                                    data[
                                                                                                                        "Medical_History_30"]))),
                                                                                                                (
                                                                                                                np.maximum(
                                                                                                                    (
                                                                                                                    data[
                                                                                                                        "Medical_History_28"]),
                                                                                                                    ((
                                                                                                                     data[
                                                                                                                         "Medical_Keyword_38"] -
                                                                                                                     data[
                                                                                                                         "Medical_History_11"])))))))))) / 2.0)) / 2.0) / 2.0)) +
         np.tanh(((data["Medical_History_31"] + (((((data["BMI"] * data["Ins_Age"]) + np.minimum(
             (data["Medical_History_13"]), (data["Product_Info_2_N"]))) / 2.0) + np.minimum(
             (np.cos(data["Product_Info_4"])), (
             (np.cos(data["Product_Info_4"]) + (data["Medical_History_1"] + data["Family_Hist_2"]))))) / 2.0)) / 2.0)) +
         ((((data["Medical_History_24"] >= data["Medical_Keyword_18"]).astype(float)) + np.minimum((np.minimum(
             ((((data["Medical_History_7"] + data["Employment_Info_2"]) / 2.0) / 2.0)),
             ((data["Medical_History_35"] * data["Medical_History_24"])))), (np.minimum((np.cos(np.round(data["Wt"]))),
                                                                                        ((data["Medical_History_5"] *
                                                                                          data[
                                                                                              "InsuredInfo_2"])))))) / 2.0) +
         np.minimum((np.sin((data["Medical_History_11"] * 2.0))), (np.maximum((data["Medical_History_4"]), ((np.maximum(
             (data["Medical_History_24"]), (data["Medical_History_15"])) + (np.maximum((data["Medical_History_24"]),
                                                                                       (data["Medical_History_15"])) + (
                                                                            (data["Wt"] + (data[
                                                                                               "Medical_History_15"] - 0.058823)) / 2.0))))))) +
         (((((data["Medical_History_23"] + (((-(data["InsuredInfo_1"])) + data["Medical_History_3"]) / 2.0)) / 2.0) + ((
                                                                                                                       (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "SumMedicalKeywords"] +
                                                                                                                       data[
                                                                                                                           "Medical_Keyword_25"]) / 2.0) + (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_Keyword_33"] + (
                                                                                                                       (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_History_17"] -
                                                                                                                       data[
                                                                                                                           "InsuredInfo_1"]) +
                                                                                                                       data[
                                                                                                                           "Medical_History_17"]) / 2.0)) / 2.0)) / 2.0)) / 2.0) / 2.0) +
         np.sin(np.minimum((data["Medical_History_33"]), ((((data["Medical_Keyword_28"] + ((((data["SumNAs"] + (
         np.cos(data["BMI"]) - data["Medical_Keyword_9"])) / 2.0) + ((((data["Medical_History_39"] + (
         data["Medical_Keyword_19"] + data["Medical_Keyword_32"])) / 2.0) + data[
                                                                          "Medical_Keyword_11"]) / 2.0)) / 2.0)) / 2.0) / 2.0)))) +
         np.minimum((np.ceil(np.minimum((((data["Employment_Info_1"] / 2.0) / 2.0)), (
         ((2.0 * np.floor(np.cos((data["Insurance_History_7"] + data["Insurance_History_5"])))) * 2.0))))), ((data[
                                                                                                                  "BMI"] * (
                                                                                                              (data[
                                                                                                                   "Medical_History_5"] >= np.cos(
                                                                                                                  data[
                                                                                                                      "Medical_History_14"])).astype(
                                                                                                                  float))))) +
         np.minimum((np.minimum((0.138462), (((data["Family_Hist_2"] <= np.round(
             (data["Medical_History_15"] + data["Product_Info_2_N"]))).astype(float))))), ((data[
                                                                                                "Medical_History_32"] + (
                                                                                            (((data[
                                                                                                   "Insurance_History_1"] < np.round(
                                                                                                np.sin(np.ceil(data[
                                                                                                                   "Product_Info_2_N"])))).astype(
                                                                                                float)) + data[
                                                                                                 "Medical_Keyword_19"]) / 2.0)))) +
         np.minimum((0.301030), (np.maximum((data["Medical_History_10"]), (((((((data["Product_Info_2_C"] >= np.round(
             (data["Product_Info_2_C"] - data["Medical_Keyword_45"]))).astype(float)) >= np.abs(((data[
                                                                                                      "Medical_Keyword_7"] < (
                                                                                                  np.cos(data[
                                                                                                             "Medical_Keyword_2"]) -
                                                                                                  data[
                                                                                                      "Medical_Keyword_26"])).astype(
             float)))).astype(float)) >= data["Medical_History_40"]).astype(float)))))) +
         np.sin(np.minimum((((data["Medical_History_13"] == data["Medical_History_18"]).astype(float))), (((((data[
                                                                                                                  "Medical_History_27"] > (
                                                                                                              data[
                                                                                                                  "Medical_History_15"] *
                                                                                                              data[
                                                                                                                  "Medical_History_18"])).astype(
             float)) * 2.0) * data["Medical_History_15"])))) +
         ((((data["Medical_History_15"] < np.minimum((np.sin(np.floor(
             np.minimum((((data["Medical_History_15"] - data["Medical_History_18"]) - data["Medical_History_18"])),
                        ((data["Medical_History_15"] - data["Medical_History_18"])))))),
                                                     (data["Medical_Keyword_41"]))).astype(float)) + (
           (data["Medical_Keyword_34"] >= np.abs(data["Medical_History_22"])).astype(float))) / 2.0) +
         (np.minimum((np.minimum((np.maximum((data["Medical_History_4"]), ((data["Medical_History_15"] / 2.0)))),
                                 (data["Medical_Keyword_43"]))), (np.sin(data["InsuredInfo_2"]))) * np.sin(
             np.minimum((np.minimum((data["Medical_History_4"]), ((data["SumNAs"] + data["Medical_History_4"])))),
                        (((data["Medical_Keyword_37"] > data["Medical_History_1"]).astype(float)))))) +
         ((0.058823 * (((data["Medical_History_6"] + np.minimum(
             ((-(np.ceil(np.maximum((data["Medical_History_19"]), (data["InsuredInfo_2"])))))),
             (data["Medical_History_22"]))) + data["Employment_Info_5"]) / 2.0)) + np.sin(
             np.cos(np.maximum((1.584910), (data["Ins_Age"]))))) +
         ((((0.058823 + ((((data["Medical_History_18"] >= data["Medical_History_28"]).astype(float)) <= data[
             "Medical_History_13"]).astype(float))) / 2.0) + (-(((((data["SumEmploymentInfo"] >= data[
             "Medical_History_18"]).astype(float)) < np.round(((0.058823 < (
         (data["Medical_Keyword_5"] < data["Medical_History_28"]).astype(float))).astype(float)))).astype(
             float))))) / 2.0) +
         ((((((data["Medical_Keyword_23"] * data["Medical_Keyword_46"]) <= np.minimum(
             ((data["Medical_Keyword_46"] - data["Medical_History_15"])), (((((data["Medical_History_20"] <= (
             data["Medical_History_5"] + data["Medical_Keyword_27"])).astype(float)) > data[
                                                                                 "Medical_History_20"]).astype(
                 float))))).astype(float)) * data["Ins_Age"]) / 2.0) / 2.0) +
         (((20.750000 <= data["Medical_History_32"]).astype(float)) + (((data["InsuredInfo_6"] * (
         (((data["Insurance_History_7"] + data["InsuredInfo_6"]) / 2.0) < data["Medical_History_23"]).astype(
             float))) * np.minimum((data["Medical_Keyword_3"]),
                                   (np.cos((1.0 / (1.0 + np.exp(- data["InsuredInfo_6"]))))))) / 2.0)) +
         np.minimum((np.minimum((((((np.minimum((data["Medical_Keyword_3"]), (data["Family_Hist_3"])) >= (
         data["Medical_Keyword_39"] * data["Medical_Keyword_6"])).astype(float)) / 2.0) / 2.0)),
                                (np.maximum((data["SumMedicalKeywords"]), (data["Medical_History_39"]))))), (
                    np.maximum((data["SumMedicalKeywords"]),
                               (((data["Medical_History_23"] + data["Medical_History_20"]) / 2.0))))) +
         np.minimum(((data["Medical_History_15"] * ((np.ceil(data["Medical_History_15"]) < np.sin(
             ((data["Medical_History_15"] > data["Employment_Info_3"]).astype(float)))).astype(float)))), ((data[
                                                                                                                "Product_Info_2_N"] * (
                                                                                                            ((
                                                                                                             np.minimum(
                                                                                                                 (data[
                                                                                                                      "Ins_Age"]),
                                                                                                                 (data[
                                                                                                                      "Medical_History_18"])) >= (
                                                                                                             1.0 / (
                                                                                                             1.0 + np.exp(
                                                                                                                 - data[
                                                                                                                     "Medical_Keyword_43"])))).astype(
                                                                                                                float)) / 2.0)))) +
         (0.094340 * (((data["Family_Hist_3"] + (
         (data["Medical_Keyword_15"] - (data["Medical_History_4"] + data["Ht"])) - ((data["SumEmploymentInfo"] >= (
         1.0 / (1.0 + np.exp(- (data["Medical_History_15"] - data["Medical_History_36"]))))).astype(
             float)))) / 2.0) - np.minimum((data["Medical_History_15"]), (data["Ins_Age"])))) +
         (-(np.sin((np.abs(data["Medical_History_5"]) * ((np.minimum(((data["Medical_History_39"] - data["BMI"])), (
         np.minimum((data["Medical_History_37"]), (np.ceil(np.ceil(data["Product_Info_4"])))))) <= (
                                                          (data["Medical_History_30"] * data["Medical_History_39"]) -
                                                          data["Medical_History_13"])).astype(float)))))) +
         np.abs((np.cos(data["Wt"]) * np.tanh((np.minimum((data["SumInsuredInfo"]), (data["Medical_History_4"])) * (
         -2.0 * (data["Medical_History_4"] * (
         data["Medical_History_10"] * ((data["Medical_History_15"] <= np.cos(-2.0)).astype(float))))))))) +
         (np.minimum((np.maximum((data["InsuredInfo_7"]), ((-(np.ceil(data["Product_Info_4"])))))), (((data["BMI"] + ((
                                                                                                                      data[
                                                                                                                          "Medical_History_12"] <= (
                                                                                                                      np.minimum(
                                                                                                                          (
                                                                                                                          (
                                                                                                                          data[
                                                                                                                              "Product_Info_4"] / 2.0)),
                                                                                                                          (
                                                                                                                          (
                                                                                                                          (
                                                                                                                          data[
                                                                                                                              "Medical_History_17"] +
                                                                                                                          data[
                                                                                                                              "InsuredInfo_7"]) / 2.0))) / 2.0)).astype(
             float))) / 2.0))) / 2.0) +
         np.sin(np.sinh(np.minimum(((1.0 / (1.0 + np.exp(- data["BMI"])))), ((data["BMI"] * np.sin(np.round(np.round(((
                                                                                                                      data[
                                                                                                                          "Medical_History_39"] - (
                                                                                                                      data[
                                                                                                                          "BMI"] * np.sin(
                                                                                                                          np.round(
                                                                                                                              data[
                                                                                                                                  "Medical_History_11"])))) + (
                                                                                                                      data[
                                                                                                                          "Insurance_History_2"] / 2.0)))))))))) +
         np.sin((data["Medical_History_5"] * ((((data["Medical_History_24"] + (
         np.maximum((data["Medical_Keyword_48"]), (data["Medical_Keyword_23"])) * data[
             "Medical_History_20"])) / 2.0) + ((data["Medical_Keyword_7"] > (
         (data["Medical_History_2"] + (1.0 / (1.0 + np.exp(- data["Medical_Keyword_8"])))) / 2.0)).astype(
             float))) / 2.0))) +
         (((data["Medical_History_20"] < ((np.ceil(data["Medical_Keyword_24"]) < (((((-(data["InsuredInfo_5"])) < data[
             "Medical_History_18"]).astype(float)) >= data["Medical_History_17"]).astype(float))).astype(
             float))).astype(float)) * (((((data["Medical_History_21"] >= data["Medical_Keyword_21"]).astype(float)) >=
                                          data["Medical_History_40"]).astype(float)) / 2.0)) +
         np.minimum(((((data["Medical_History_15"] + data["InsuredInfo_1"]) / 2.0) * (
         np.ceil(data["Medical_History_23"]) * 0.058823))), (((data["Medical_History_1"] + (
         (data["InsuredInfo_2"] <= (data["Medical_History_37"] * 2.0)).astype(float))) / 2.0))) +
         np.sin((np.ceil(np.ceil(np.minimum((np.round(((
                                                       data["Medical_Keyword_12"] + np.minimum((data["Product_Info_4"]),
                                                                                               (data[
                                                                                                    "Medical_Keyword_15"]))) / 2.0))),
                                            (data["Medical_History_40"])))) * ((
                                                                               data["Medical_Keyword_15"] <= np.minimum(
                                                                                   (data["Medical_History_4"]), ((data[
                                                                                                                      "InsuredInfo_4"] -
                                                                                                                  data[
                                                                                                                      "BMI"])))).astype(
             float)))) +
         np.minimum(((0.058823 * (
         np.cos(data["Employment_Info_4"]) - np.minimum((data["Medical_History_3"]), (np.sinh(data["Ins_Age"])))))), (((
                                                                                                                       (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_Keyword_11"] >=
                                                                                                                       data[
                                                                                                                           "Product_Info_2_C"]).astype(
                                                                                                                           float)) + (
                                                                                                                       -(
                                                                                                                       (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_History_38"] >= 0.0).astype(
                                                                                                                           float))))) / 2.0))) +
         np.sin((data["Medical_History_11"] * ((data["Medical_History_37"] >= (
         ((data["Medical_Keyword_25"] <= (data["Medical_History_8"] * data["Medical_Keyword_33"])).astype(float)) - (((
                                                                                                                      data[
                                                                                                                          "Medical_Keyword_16"] +
                                                                                                                      data[
                                                                                                                          "Medical_Keyword_5"]) + (
                                                                                                                      (
                                                                                                                      data[
                                                                                                                          "Medical_Keyword_9"] +
                                                                                                                      data[
                                                                                                                          "Medical_Keyword_29"]) -
                                                                                                                      data[
                                                                                                                          "Medical_History_18"])) / 2.0))).astype(
             float)))) +
         (np.sin(np.sin(np.minimum((data["SumMedicalKeywords"]), (data["InsuredInfo_5"])))) * ((data[
                                                                                                    "SumInsuranceHistory"] >= np.ceil(
             ((np.minimum((data["Medical_History_17"]), (((((data["Medical_History_17"] >= np.abs(
                 data["InsuredInfo_5"])).astype(float)) >= np.abs(data["SumMedicalKeywords"])).astype(float)))) + data[
                   "Medical_Keyword_19"]) / 2.0))).astype(float))) +
         np.minimum((((np.tanh(np.cos(((data["Ht"] + np.minimum((data["Medical_Keyword_39"]),
                                                                (data["Medical_Keyword_47"]))) / 2.0))) < np.minimum(
             (data["Medical_Keyword_14"]), (((data["Medical_Keyword_39"] + data["Medical_Keyword_34"]) / 2.0)))).astype(
             float))), (np.cos(
             np.minimum((data["Product_Info_7"]), (((data["SumNAs"] + data["Medical_Keyword_42"]) / 2.0)))))) +
         (0.058823 * np.round((((data["Medical_History_1"] >= (data["Medical_Keyword_47"] * 2.0)).astype(
             float)) - np.abs(np.sinh(((data["Product_Info_2_N"] + np.sinh(np.minimum((data["Product_Info_2_N"]), (
         ((data["Medical_Keyword_11"] + data["Medical_Keyword_47"]) / 2.0))))) / 2.0)))))) +
         np.minimum((0.138462), (np.maximum((((data["Medical_Keyword_21"] + data["Medical_Keyword_16"]) / 2.0)), (
         np.maximum((np.maximum((((data["Medical_Keyword_20"] + data["Medical_Keyword_2"]) / 2.0)), (
         ((((data["Medical_Keyword_29"] + data["Product_Info_7"]) / 2.0) + data["Medical_Keyword_8"]) / 2.0)))),
                    (((0.058823 >= np.cos(data["Family_Hist_5"])).astype(float)))))))) +
         np.tanh(np.tanh((data["Medical_Keyword_3"] * ((((data["Product_Info_5"] >= data["Medical_History_7"]).astype(
             float)) >= (((data["Medical_History_22"] + (
         (np.ceil(data["Product_Info_4"]) - data["Product_Info_5"]) - data["Medical_History_35"])) / 2.0) - data[
                             "Medical_Keyword_38"])).astype(float))))) +
         np.minimum(((np.maximum((np.ceil(data["Medical_History_6"])), ((data["Ht"] / 2.0))) / 2.0)), (((
                                                                                                        2.718282 < np.maximum(
                                                                                                            ((data[
                                                                                                                  "SumNAs"] - (
                                                                                                              (data[
                                                                                                                   "Medical_History_28"] >
                                                                                                               data[
                                                                                                                   "SumEmploymentInfo"]).astype(
                                                                                                                  float)))),
                                                                                                            (((data[
                                                                                                                   "Medical_History_19"] + (
                                                                                                               data[
                                                                                                                   "Medical_History_28"] *
                                                                                                               data[
                                                                                                                   "Medical_History_7"])) / 2.0)))).astype(
             float)))) +
         np.tanh((np.minimum((data["Medical_Keyword_24"]), (data["SumMedicalKeywords"])) * ((data[
                                                                                                 "Medical_History_10"] >= (
                                                                                             np.cos(data[
                                                                                                        "InsuredInfo_1"]) * (
                                                                                             np.minimum((data[
                                                                                                             "Medical_History_27"]),
                                                                                                        (((data[
                                                                                                               "Medical_History_18"] >=
                                                                                                           data[
                                                                                                               "Medical_Keyword_24"]).astype(
                                                                                                            float)))) *
                                                                                             data[
                                                                                                 "Medical_History_18"]))).astype(
             float)))) +
         ((np.minimum((np.ceil(np.minimum((data["Product_Info_7"]), (data["Medical_History_6"])))),
                      (data["Medical_Keyword_21"])) > np.cos(((np.minimum((data["Medical_History_10"]), ((data[
                                                                                                              "Medical_History_30"] - np.minimum(
             (data["Medical_History_22"]), (data["Medical_History_40"]))))) * 2.0) - np.minimum(
             (data["Product_Info_7"]), (data["Medical_History_40"]))))).astype(float)) +
         np.ceil(np.minimum((data["Medical_Keyword_13"]), ((((data["Medical_Keyword_38"] + (
         data["Medical_Keyword_11"] * (-(((data["Medical_Keyword_43"] >= (data["Medical_History_7"] * (
         data["Medical_History_22"] + (data["Medical_History_3"] * np.floor(data["Medical_History_28"]))))).astype(
             float)))))) / 2.0) / 2.0)))) +
         (np.sin(np.sin(np.floor(np.sin(((data["Medical_History_20"] + ((np.abs(
             np.maximum((np.abs(data["Medical_History_27"])), (data["Medical_Keyword_31"]))) + np.minimum(
             (data["Medical_History_24"]), ((((data["Medical_Keyword_4"] * data["Medical_History_22"]) + data[
                 "Medical_Keyword_31"]) / 2.0)))) / 2.0)) / 2.0))))) * 2.0) +
         ((np.maximum((data["Medical_History_28"]), (np.maximum((data["Medical_History_27"]), (((np.cos(
             np.maximum((np.maximum((data["Medical_Keyword_23"]), (data["Wt"]))), (data["Medical_Keyword_17"]))) > (
                                                                                                 np.maximum((data[
                                                                                                                 "Medical_History_19"]),
                                                                                                            (data[
                                                                                                                 "Medical_Keyword_24"])) / 2.0)).astype(
             float)))))) < (data["Medical_Keyword_32"] * data["Medical_Keyword_46"])).astype(float)) +
         (((np.minimum((((data["SumMedicalKeywords"] > data["Medical_History_22"]).astype(float))), (((np.minimum(
             (((data["Medical_Keyword_3"] > data["Medical_History_22"]).astype(float))),
             (((data["Medical_Keyword_40"] > data["Medical_History_31"]).astype(float)))) > data[
                                                                                                           "Medical_History_31"]).astype(
             float)))) >= np.maximum((data["Medical_History_20"]),
                                     (((data["Insurance_History_9"] < data["InsuredInfo_1"]).astype(float))))).astype(
             float)) / 2.0) +
         (((np.minimum((np.sin(np.maximum((data["SumMedicalKeywords"]), (np.ceil(
             np.maximum(((-(data["Medical_History_12"]))),
                        (np.maximum(((data["Employment_Info_2"] * 5.200000)), (data["Medical_Keyword_3"]))))))))), (((
                                                                                                                     data[
                                                                                                                         "Medical_History_37"] > np.abs(
                                                                                                                         data[
                                                                                                                             "SumMedicalKeywords"])).astype(
             float)))) / 2.0) / 2.0) / 2.0) +
         np.floor(np.sin(np.round((data["Product_Info_1"] * ((((((data["Medical_History_12"] > (
         data["Medical_Keyword_41"] * data["Insurance_History_3"])).astype(float)) > np.cos(
             np.abs(data["Employment_Info_1"]))).astype(float)) != (data["Medical_History_8"] * (
         (data["Medical_Keyword_14"] >= data["Medical_History_7"]).astype(float)))).astype(float)))))) +
         np.minimum((np.cos(data["SumEmploymentInfo"])), (np.minimum((np.sin(np.round(data["Medical_History_35"]))), (
         np.cos(np.minimum((data["Medical_History_10"]), ((data["Medical_Keyword_42"] + np.maximum(
             (data["Medical_History_3"]),
             (np.maximum((np.floor(data["SumMedicalHistory"])), (data["Medical_History_35"])))))))))))) +
         np.sin(np.maximum(((0.058823 * ((np.ceil(data["Product_Info_2_C"]) < data["Ins_Age"]).astype(float)))), (((
                                                                                                                   np.minimum(
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "InsuredInfo_5"]),
                                                                                                                       (
                                                                                                                       (
                                                                                                                       0.058823 + (
                                                                                                                       (
                                                                                                                       (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_History_4"] >
                                                                                                                       data[
                                                                                                                           "InsuredInfo_6"]).astype(
                                                                                                                           float)) > np.abs(
                                                                                                                           data[
                                                                                                                               "Ins_Age"])).astype(
                                                                                                                           float))))) * 2.0) * 2.0)))) +
         ((data["InsuredInfo_5"] * (((data["InsuredInfo_5"] * data["InsuredInfo_1"]) >= (
         (data["SumInsuredInfo"] >= data["Medical_Keyword_24"]).astype(float))).astype(float))) * (
          (((data["Medical_History_40"] <= np.cos((data["SumInsuredInfo"] * 2.0))).astype(float)) / 2.0) / 2.0)) +
         (np.minimum((np.maximum((((np.round(data["Ins_Age"]) + 0.693147) / 2.0)),
                                 ((data["Medical_History_33"] - data["InsuredInfo_7"])))), ((((data[
                                                                                                   "Medical_Keyword_46"] >= np.maximum(
             (data["Ins_Age"]), (data["Medical_History_39"]))).astype(float)) * data["InsuredInfo_6"]))) / 2.0) +
         np.minimum(((-(((data["Product_Info_3"] <= np.minimum((-3.0), (
         np.minimum(((data["Ht"] + (-3.0 * data["InsuredInfo_3"]))), ((data["Medical_History_29"] + -3.0)))))).astype(
             float))))), (np.maximum((data["Medical_History_11"]), (np.tanh(data["Product_Info_3"]))))) +
         np.sin(((1.197370 <= np.minimum((np.abs(data["Employment_Info_2"])), (((((data["InsuredInfo_1"] + np.minimum(
             (data["Medical_Keyword_38"]), (data["Medical_History_28"]))) + ((data["Medical_Keyword_30"] > data[
             "Medical_Keyword_7"]).astype(float))) + np.minimum((data["Medical_Keyword_1"]),
                                                                (data["Medical_Keyword_16"]))) / 2.0)))).astype(
             float))) +
         ((np.minimum((data["Medical_Keyword_47"]), (np.maximum((data["Medical_History_5"]), (
         np.maximum((data["Medical_Keyword_2"]), (np.maximum((data["Medical_Keyword_43"]), (
         np.maximum((data["Medical_Keyword_26"]), ((data["Medical_Keyword_36"] + data["Product_Info_5"]))))))))))) >= (
           np.abs(data["Medical_Keyword_14"]) - np.minimum((data["Medical_History_28"]),
                                                           (data["Medical_Keyword_36"])))).astype(float)) +
         (0.094340 * np.floor(np.cos((((data["Medical_History_18"] + (data["Medical_History_13"] + (
         data["Medical_Keyword_40"] * ((np.tanh(data["Medical_History_10"]) != (
         (data["Product_Info_2_C"] == data["Product_Info_2_C"]).astype(float))).astype(float))))) / 2.0) * 2.0)))) +
         np.minimum((0.094340), ((((((data["Product_Info_2_C"] > (((((data["Medical_Keyword_22"] + data[
             "Medical_History_37"]) / 2.0) + ((data["Medical_Keyword_21"] + data["Product_Info_6"]) / 2.0)) > data[
                                                                       "Medical_Keyword_10"]).astype(float))).astype(
             float)) + np.maximum((data["Medical_History_10"]), (data["Medical_Keyword_11"]))) / 2.0) / 2.0))) +
         ((((data["Employment_Info_2"] + (-(((2.409090 < data["Ins_Age"]).astype(float))))) <= data[
             "Product_Info_4"]).astype(float)) * np.sinh(
             np.ceil(np.minimum(((-(((2.409090 < data["Ins_Age"]).astype(float))))), ((data["Wt"] / 2.0)))))) +
         np.ceil((((np.tanh(data["Medical_Keyword_42"]) >= (
         (data["Medical_Keyword_20"] < data["Medical_History_11"]).astype(float))).astype(float)) - ((np.minimum(
             (data["Medical_History_23"]), (data["Wt"])) > ((((data["Insurance_History_9"] <= data[
             "Medical_History_10"]).astype(float)) >= np.minimum((data["Medical_Keyword_38"]), (data["Ht"]))).astype(
             float))).astype(float)))) +
         (data["Insurance_History_1"] * (-((data["Medical_Keyword_39"] * (np.round(np.sin((data[
                                                                                               "Medical_History_38"] - np.minimum(
             (((data["Medical_History_9"] > ((data["Medical_Keyword_39"] > data["SumNAs"]).astype(float))).astype(
                 float))), (data["SumMedicalKeywords"]))))) / 2.0))))) +
         ((((data["Insurance_History_1"] >= data["Medical_History_5"]).astype(float)) < ((((
                                                                                           data["Medical_Keyword_43"] >=
                                                                                           data[
                                                                                               "Medical_History_7"]).astype(
             float)) > (((data["Medical_Keyword_19"] + ((data["Medical_Keyword_43"] >= data["Wt"]).astype(float))) <= (
         (data["Product_Info_6"] >= data["Insurance_History_1"]).astype(float))).astype(float))).astype(float))).astype(
             float)) +
         np.minimum((np.maximum(((data["Medical_History_27"] - (
         data["Medical_Keyword_30"] * (data["Medical_History_7"] - data["Medical_History_19"])))),
                                (data["Medical_Keyword_25"]))), ((data["Medical_Keyword_25"] * (
         data["Medical_Keyword_35"] * ((data["Medical_Keyword_12"] >= data["Medical_Keyword_18"]).astype(float)))))) +
         (((data["Wt"] / 2.0) * (((data["Medical_Keyword_23"] * ((data["Medical_History_31"] <= ((((data[
                                                                                                        "Medical_History_30"] +
                                                                                                    data[
                                                                                                        "Medical_Keyword_31"]) / 2.0) +
                                                                                                  data[
                                                                                                      "Medical_Keyword_46"]) / 2.0)).astype(
             float))) <= ((data["Medical_History_30"] + data["Medical_Keyword_14"]) / 2.0)).astype(float))) / 2.0) +
         np.sin(np.maximum((data["Medical_Keyword_43"]), (((((((data["Employment_Info_1"] < 0.693147).astype(float)) < (
         (data["Medical_Keyword_35"] > data["Medical_History_10"]).astype(float))).astype(float)) == np.maximum(
             (data["Medical_History_27"]), ((((data["Medical_History_22"] * data["SumMedicalHistory"]) <= data[
                 "Employment_Info_4"]).astype(float))))).astype(float))))) +
         ((((np.minimum((np.floor(((data["Medical_History_14"] >= data["Insurance_History_5"]).astype(float)))), (
         np.minimum((((data["Medical_Keyword_16"] + np.ceil(data["Product_Info_7"])) / 2.0)),
                    (np.floor((-(data["Insurance_History_5"]))))))) == (
             (data["Medical_History_21"] >= data["Medical_History_24"]).astype(float))).astype(float)) > np.cos(
             data["Medical_History_5"])).astype(float)) +
         ((2.409090 < ((data["SumMedicalHistory"] + (np.minimum((data["Medical_History_10"]), (((data[
                                                                                                     "Medical_Keyword_14"] + (
                                                                                                 ((np.abs((data[
                                                                                                               "Medical_Keyword_15"] *
                                                                                                           data[
                                                                                                               "Medical_History_36"])) +
                                                                                                   data[
                                                                                                       "Medical_Keyword_20"]) +
                                                                                                  data[
                                                                                                      "Medical_Keyword_4"]) / 2.0)) / 2.0))) * 2.0)) / 2.0)).astype(
             float)))
    return p + 4.5


def Individual4(data):
    p = ((np.minimum((data["Medical_History_40"]),
                     (np.power(np.abs((-2.0 + data["BMI"])), data["Medical_History_39"]))) + np.power(
        np.abs((-2.0 + data["BMI"])), ((data["Medical_History_4"] + (
        data["Product_Info_4"] + (((data["Medical_History_4"] * 2.0) + data["Medical_History_23"]) / 2.0))) / 2.0))) +
         (np.sin(data["Medical_History_30"]) - (np.maximum(
             (((np.power(np.abs(data["Product_Info_4"]), 31.006277) <= data["Medical_Keyword_3"]).astype(float))), ((((
                                                                                                                      np.abs(
                                                                                                                          data[
                                                                                                                              "Medical_History_15"]) <= np.minimum(
                                                                                                                          (
                                                                                                                          data[
                                                                                                                              "SumInsuranceHistory"]),
                                                                                                                          (
                                                                                                                          data[
                                                                                                                              "Medical_History_15"]))).astype(
                 float)) * 2.0))) + ((data["Medical_History_13"] >= data["Medical_History_15"]).astype(float)))) +
         (np.ceil(((data["Medical_History_39"] < np.floor(data["InsuredInfo_6"])).astype(float))) - ((data[
                                                                                                          "Medical_History_20"] < np.maximum(
             (np.floor((data["InsuredInfo_5"] * data["Insurance_History_2"]))), (np.sinh(
                 np.maximum((np.maximum((data["InsuredInfo_7"]), (data["Medical_History_5"]))),
                            ((data["Ins_Age"] * 2.0))))))).astype(float))) +
         (np.minimum((data["Medical_History_18"]), ((((np.tanh(
             np.minimum((data["Medical_History_24"]), ((data["Medical_History_6"] - data["Medical_History_15"])))) + ((
                                                                                                                      data[
                                                                                                                          "Medical_History_18"] <= (
                                                                                                                      (
                                                                                                                      data[
                                                                                                                          "Medical_History_13"] >
                                                                                                                      data[
                                                                                                                          "Medical_History_35"]).astype(
                                                                                                                          float))).astype(
             float))) / 2.0) - data["Medical_Keyword_38"]))) - (
          (data["Employment_Info_2"] < data["Medical_History_28"]).astype(float))) +
         (((data["Medical_Keyword_41"] + np.sin((data["BMI"] / 2.0))) / 2.0) - (((-(
         ((data["Medical_History_27"] <= data["Medical_History_35"]).astype(float)))) < ((data["Medical_History_7"] <= (
         data["Medical_History_5"] + (
         ((data["BMI"] / 2.0) <= np.sin(data["Medical_History_11"])).astype(float)))).astype(float))).astype(float))) +
         (((-(np.maximum((data["Medical_Keyword_48"]), (np.floor(data["Product_Info_4"]))))) + ((np.maximum(
             (data["Family_Hist_4"]), (np.sin((data["InsuredInfo_6"] - data["InsuredInfo_1"])))) + (((data[
                                                                                                          "Insurance_History_5"] + (
                                                                                                      data[
                                                                                                          "InsuredInfo_6"] -
                                                                                                      data[
                                                                                                          "InsuredInfo_1"])) +
                                                                                                     data[
                                                                                                         "Family_Hist_2"]) / 2.0)) / 2.0)) / 2.0) +
         np.sin(np.minimum((np.cos(data["Medical_History_19"])), ((((data["Medical_History_40"] <= (
         (data["Medical_History_31"] <= data["Medical_History_39"]).astype(float))).astype(float)) - (
                                                                   data["Medical_Keyword_9"] + (((data[
                                                                                                      "Medical_History_31"] <=
                                                                                                  data[
                                                                                                      "InsuredInfo_2"]).astype(
                                                                       float)) - ((data["Medical_Keyword_48"] >= data[
                                                                       "Medical_History_39"]).astype(float)))))))) +
         ((((data["Medical_Keyword_33"] + ((((data["Medical_Keyword_25"] - np.sin(np.sin(data["Product_Info_2_N"]))) + (
         (round(2.675680) < data["Medical_History_1"]).astype(float))) / 2.0) - data["Medical_Keyword_23"])) / 2.0) + (
           data["Medical_History_17"] - np.round(2.675680))) / 2.0) +
         (((np.sin((((data["Medical_Keyword_45"] + (
         np.sin(np.sinh(data["Product_Info_2_C"])) + data["Medical_Keyword_22"])) / 2.0) * 2.0)) + ((data[
                                                                                                         "Employment_Info_1"] > (
                                                                                                     (data[
                                                                                                          "Medical_History_4"] > (
                                                                                                      (np.minimum((data[
                                                                                                                       "Medical_History_4"]),
                                                                                                                  (data[
                                                                                                                       "Medical_History_15"])) <=
                                                                                                       data[
                                                                                                           "Medical_Keyword_45"]).astype(
                                                                                                          float))).astype(
                                                                                                         float))).astype(
             float))) / 2.0) / 2.0) +
         np.minimum((((np.round(((data["BMI"] > 0.367879).astype(float))) - data["BMI"]) * data["Product_Info_4"])), (
         np.minimum((((data["Insurance_History_7"] < data["Product_Info_2_N"]).astype(float))),
                    (((data["BMI"] + np.floor((data["Product_Info_4"] - 0.058823))) / 2.0))))) +
         np.minimum((np.sin((data["Medical_History_35"] + np.power(np.abs(-2.0), data["Medical_History_11"])))), ((
                                                                                                                  np.sin(
                                                                                                                      ((
                                                                                                                       (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_History_39"] <= (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_Keyword_3"] <
                                                                                                                       data[
                                                                                                                           "SumMedicalKeywords"]).astype(
                                                                                                                           float))).astype(
                                                                                                                           float)) + (
                                                                                                                       -(
                                                                                                                       (
                                                                                                                       (
                                                                                                                       data[
                                                                                                                           "Medical_History_39"] < np.sinh(
                                                                                                                           data[
                                                                                                                               "Medical_Keyword_3"])).astype(
                                                                                                                           float))))) / 2.0)) * 2.0))) +
         ((((-(((((data["SumInsuranceHistory"] - data["Medical_History_15"]) < data["Medical_Keyword_37"]).astype(
             float)) / 2.0))) - ((data["Medical_Keyword_37"] + data["Employment_Info_1"]) / 2.0)) + np.floor(
             np.power(np.abs(data["Medical_Keyword_41"]),
                      np.power(np.abs(data["BMI"]), data["Medical_History_15"])))) / 2.0) +
         (np.minimum(((0.058823 * np.cos(data["SumEmploymentInfo"]))), (
         np.minimum(((((data["Medical_History_39"] < data["Product_Info_2_N"]).astype(float)) / 2.0)), (
         np.minimum((data["Employment_Info_1"]), (((data["BMI"] * 2.0) - np.sinh(data["Ins_Age"])))))))) * 2.0) +
         ((((((0.720430 <= data["Ins_Age"]).astype(float)) * data["BMI"]) + np.power(
             np.abs(np.round((data["Ins_Age"] * np.sinh(data["BMI"])))), data["Medical_History_11"])) / 2.0) - ((data[
                                                                                                                     "Medical_History_23"] <= (
                                                                                                                 (data[
                                                                                                                      "Medical_Keyword_16"] >=
                                                                                                                  data[
                                                                                                                      "SumMedicalKeywords"]).astype(
                                                                                                                     float))).astype(
             float))) +
         np.minimum((data["Medical_Keyword_38"]), ((data["Medical_Keyword_3"] + np.minimum(
             (((-(data["Medical_Keyword_18"])) / 2.0)), (np.sinh(np.floor(np.minimum((data["Employment_Info_1"]), (
             np.floor(np.sinh(
                 np.floor((data["Insurance_History_5"] * np.round(np.cos(data["Insurance_History_7"]))))))))))))))) +
         ((((data["BMI"] <= (
         np.cos((data["Medical_History_4"] * (1.0 / (1.0 + np.exp(- data["Medical_History_11"]))))) / 2.0)).astype(
             float)) + (((data["Medical_Keyword_19"] + (
         -(((data["Product_Info_4"] < data["Medical_Keyword_24"]).astype(float))))) + (
                         (data["Product_Info_4"] >= data["Medical_History_4"]).astype(float))) / 2.0)) / 2.0) +
         np.minimum((np.cos(((((data["Medical_History_20"] >= data["SumMedicalKeywords"]).astype(float)) + data[
             "Medical_History_30"]) / 2.0))), (
                    np.minimum((((-(np.power(np.abs(data["Ht"]), data["SumEmploymentInfo"]))) / 2.0)), ((-(((data[
                                                                                                                 "Ins_Age"] >= np.sin(
                        ((data["Medical_History_20"] > data["Medical_Keyword_31"]).astype(float)))).astype(
                        float)))))))) +
         (1.0 / (1.0 + np.exp(- (np.maximum((data["Insurance_History_2"]), (data["Insurance_History_1"])) - (
         np.power(np.abs(data["Medical_History_11"]), np.power(np.abs(data["Insurance_History_1"]),
                                                               np.maximum((np.round(data["BMI"])), (
                                                               np.maximum((data["Ins_Age"]), (
                                                               data["SumMedicalKeywords"])))))) - np.minimum(
             (data["SumMedicalKeywords"]), (data["Family_Hist_3"]))))))) +
         np.minimum((((((data["SumMedicalKeywords"] + data["Medical_History_15"]) / 2.0) + np.cos(
             data["InsuredInfo_5"])) / 2.0)), ((-(np.power(np.abs(data["Wt"]), np.maximum((data["Medical_History_23"]),
                                                                                          (((data[
                                                                                                 "SumMedicalKeywords"] + (
                                                                                             8.0 * ((data[
                                                                                                         "Medical_History_15"] < 8.0).astype(
                                                                                                 float)))) / 2.0)))))))) +
         np.minimum((np.power(np.abs(data["BMI"]), data["Medical_History_23"])), (((((np.power(
             np.abs(data["Medical_History_15"]), data["Product_Info_3"]) >= (data["Medical_Keyword_28"] + np.round(
             data["BMI"]))).astype(float)) < np.maximum((data["Medical_History_5"]), (data["InsuredInfo_7"]))).astype(
             float)))) +
         np.minimum(((data["Medical_Keyword_38"] + (data["Ins_Age"] * (-(((data["Medical_Keyword_47"] + (
         (data["InsuredInfo_6"] < data["Medical_History_18"]).astype(float))) / 2.0)))))), ((-(((data[
                                                                                                     "Medical_History_20"] <= (
                                                                                                 (data[
                                                                                                      "InsuredInfo_1"] + (
                                                                                                  (data[
                                                                                                       "Insurance_History_2"] +
                                                                                                   data[
                                                                                                       "Medical_Keyword_47"]) / 2.0)) / 2.0)).astype(
             float)))))) +
         (((data["Medical_History_21"] > data["Medical_History_7"]).astype(float)) + ((-(((1.414214 < (
         data["Medical_History_5"] * (
         ((data["Medical_History_15"] < (data["Ins_Age"] - data["Medical_History_28"])).astype(float)) - data[
             "Ins_Age"]))).astype(float)))) * 2.0)) +
         (((0.138462 * np.sin(((-(0.138462)) * (data["Medical_History_2"] / 2.0)))) + ((data["Medical_History_39"] < ((
                                                                                                                      data[
                                                                                                                          "Medical_History_30"] + (
                                                                                                                      ((
                                                                                                                       data[
                                                                                                                           "Medical_Keyword_3"] >= 0.138462).astype(
                                                                                                                          float)) * np.maximum(
                                                                                                                          (
                                                                                                                          data[
                                                                                                                              "Medical_Keyword_24"]),
                                                                                                                          (
                                                                                                                          data[
                                                                                                                              "Medical_Keyword_40"])))) / 2.0)).astype(
             float))) / 2.0) +
         (((((data["Medical_History_39"] < data["Medical_History_41"]).astype(float)) + (np.tanh(((((data[
                                                                                                         "Medical_History_1"] > np.maximum(
             (np.minimum((9.869604), ((data["Medical_History_39"] / 2.0)))), (9.869604))).astype(float)) + data[
                                                                                                       "Medical_Keyword_34"]) / 2.0)) - (
                                                                                         (data["Medical_History_39"] <
                                                                                          data["InsuredInfo_6"]).astype(
                                                                                             float)))) / 2.0) / 2.0) +
         (((np.minimum((data["Medical_Keyword_39"]), (data["Medical_Keyword_47"])) + np.power(np.abs(((data[
                                                                                                           "Medical_Keyword_14"] > (
                                                                                                       -(np.minimum((
                                                                                                                    data[
                                                                                                                        "Medical_History_10"]),
                                                                                                                    ((
                                                                                                                     data[
                                                                                                                         "Medical_Keyword_37"] +
                                                                                                                     data[
                                                                                                                         "Medical_Keyword_24"])))))).astype(
             float))), np.sinh((data["Medical_History_40"] + np.maximum((data["Medical_History_1"]),
                                                                        (data["Family_Hist_2"])))))) / 2.0) * 2.0) +
         np.minimum((np.cos((data["BMI"] * 2.0))), ((np.minimum(
             (np.cos((data["Medical_History_4"] - np.sinh(data["BMI"])))), (((data["Medical_History_4"] > (
             np.sinh(data["BMI"]) + np.floor(np.ceil((data["BMI"] * 2.0))))).astype(float)))) * 2.0))) +
         (0.138462 * (((((data["SumMedicalKeywords"] < np.cos(data["Product_Info_2_C"])).astype(float)) >= np.sin(
             ((data["Product_Info_3"] + data["SumMedicalKeywords"]) / 2.0))).astype(float)) - ((data[
                                                                                                    "Medical_History_24"] >= (
                                                                                                -((data[
                                                                                                       "Medical_History_4"] * np.sin(
                                                                                                    data[
                                                                                                        "Medical_Keyword_23"]))))).astype(
             float)))) +
         (np.maximum(((0.318310 * ((data["BMI"] < np.minimum(
             ((1.0 / (1.0 + np.exp(- np.sin(data["Medical_History_32"]))))),
             ((np.sin(data["Medical_History_40"]) * 2.0)))).astype(float)))),
                     (((data["Medical_History_18"] > data["Medical_History_40"]).astype(float)))) - (
          (9.869604 > data["Product_Info_3"]).astype(float))))
    return p


def Individual5(data):
    p = ((5.428570 - ((data["BMI"] + (((data["BMI"] + (
    ((data["Medical_Keyword_3"] - (data["Product_Info_4"] - data["SumMedicalKeywords"])) - np.cos(data["BMI"])) - data[
        "Medical_History_23"])) / 2.0) - np.sin(data["Medical_History_4"]))) / 2.0)) +
         (np.tanh(((data["Medical_History_40"] + (data["Medical_History_15"] - ((((data["InsuredInfo_5"] + data[
             "Medical_History_5"]) / 2.0) + (data["Insurance_History_2"] + (
         data["Medical_History_30"] - data["Medical_History_4"]))) / 2.0))) / 2.0)) - (
          (np.ceil(data["Product_Info_4"]) < data["InsuredInfo_7"]).astype(float))) +
         np.tanh((np.abs(data["Medical_Keyword_41"]) + (
         data["Medical_History_27"] + np.minimum((data["Medical_History_20"]), (((data["BMI"] + (((data[
                                                                                                       "Family_Hist_4"] + (
                                                                                                   (data[
                                                                                                        "Medical_History_39"] +
                                                                                                    data[
                                                                                                        "Medical_History_13"]) +
                                                                                                   data[
                                                                                                       "InsuredInfo_6"])) / 2.0) +
                                                                                                 data[
                                                                                                     "Medical_History_11"])) / 2.0)))))) +
         ((((data["Medical_Keyword_38"] < data["Medical_History_24"]).astype(float)) + ((np.minimum(
             (data["Insurance_History_5"]), (((np.minimum(
                 ((-(np.maximum((data["Medical_History_18"]), (data["Medical_Keyword_38"]))))),
                 (np.minimum((data["Medical_History_40"]), (data["Medical_History_7"])))) + data[
                                                   "SumMedicalKeywords"]) / 2.0))) + ((data["InsuredInfo_5"] < data[
             "Product_Info_2_N"]).astype(float))) / 2.0)) / 2.0) +
         np.minimum(((1.414214 - data["Ins_Age"])), (
         np.minimum((np.cos((data["Medical_History_28"] * data["Employment_Info_3"]))), (
         np.minimum((np.cos(np.floor(data["Product_Info_4"]))), (np.tanh((np.tanh(data["Medical_History_31"]) - (
         (data["Medical_History_5"] + data["Medical_History_35"]) / 2.0))))))))) +
         ((data["BMI"] * (((((data["Ins_Age"] + ((-(data["Medical_History_4"])) * 2.0)) / 2.0) + (
         ((-(data["Wt"])) + (data["SumMedicalKeywords"] * 2.0)) / 2.0)) / 2.0) / 2.0)) / 2.0) +
         ((np.minimum((np.cos(data["InsuredInfo_1"])), (np.tanh((-(data["InsuredInfo_5"]))))) + ((np.tanh(
             data["Medical_History_1"]) + (((np.maximum((data["Medical_Keyword_25"]), (
         (np.maximum((data["Medical_Keyword_6"]), (data["Medical_Keyword_45"])) + data["InsuredInfo_6"]))) + data[
                                                 "Medical_Keyword_22"]) / 2.0) / 2.0)) / 2.0)) / 2.0) +
         np.sin(np.round(np.minimum(((np.minimum((data["Medical_History_11"]), (np.minimum((data["Medical_History_40"]),
                                                                                           (np.minimum((data[
                                                                                                            "Medical_History_11"]),
                                                                                                       (data[
                                                                                                            "Medical_History_17"])))))) * 2.0)),
                                    (np.minimum((data["Medical_History_33"]), ((-(((data["InsuredInfo_2"] > np.tanh(
                                        (data["Medical_History_35"] * data["Medical_Keyword_9"]))).astype(
                                        float)))))))))) +
         np.sin(np.maximum((np.minimum((np.cos((data["Medical_Keyword_3"] - data["Medical_Keyword_48"]))), (
         np.minimum((np.cos(data["SumEmploymentInfo"])), (((np.tanh(np.sin(np.ceil(data["Medical_History_15"]))) + (
         (data["Product_Info_2_C"] >= np.cos(data["Medical_Keyword_33"])).astype(float))) / 2.0)))))),
                           (data["Medical_Keyword_33"]))) +
         np.sin(np.minimum((((data["Product_Info_2_N"] + np.abs(data["Product_Info_2_C"])) / 2.0)), (
         np.minimum((0.367879), (
         np.maximum((data["Medical_Keyword_29"]), ((data["Medical_Keyword_38"] * data["Medical_History_30"])))))))) +
         (0.058823 * (((data["Medical_Keyword_3"] * (data["SumMedicalKeywords"] + np.floor(
             np.minimum(((data["Medical_Keyword_15"] + data["Family_Hist_5"])), (data["Medical_History_30"]))))) + (
                       data["Family_Hist_2"] + data["Family_Hist_5"])) + (
                      (data["SumMedicalKeywords"] + data["Medical_History_33"]) / 2.0))) +
         np.minimum((((data["Product_Info_4"] < np.floor(data["InsuredInfo_7"])).astype(float))), (((((data[
                                                                                                           "Medical_History_20"] > (
                                                                                                       (data[
                                                                                                            "Product_Info_4"] < np.tanh(
                                                                                                           (((np.tanh(
                                                                                                               np.tanh(
                                                                                                                   data[
                                                                                                                       "Product_Info_4"])) + np.floor(
                                                                                                               np.floor(
                                                                                                                   data[
                                                                                                                       "InsuredInfo_7"]))) / 2.0) * 2.0))).astype(
                                                                                                           float))).astype(
             float)) + data["BMI"]) / 2.0))) +
         np.tanh(np.tanh((np.minimum((data["BMI"]),
                                     (((data["Medical_History_40"] + data["Medical_History_15"]) / 2.0))) * np.minimum((
                                                                                                                       np.tanh(
                                                                                                                           (
                                                                                                                           (
                                                                                                                           (
                                                                                                                           1.0 / (
                                                                                                                           1.0 + np.exp(
                                                                                                                               -
                                                                                                                               data[
                                                                                                                                   "Medical_History_15"]))) > (
                                                                                                                           -(
                                                                                                                           np.tanh(
                                                                                                                               data[
                                                                                                                                   "Medical_History_15"])))).astype(
                                                                                                                               float)))),
                                                                                                                       (
                                                                                                                       (
                                                                                                                       -(
                                                                                                                       np.tanh(
                                                                                                                           data[
                                                                                                                               "Medical_History_15"])))))))) +
         ((((data["Ins_Age"] * ((data["Medical_History_33"] <= (data["Ins_Age"] * (
         (((data["Medical_History_38"] >= data["Medical_History_38"]).astype(float)) <= data["BMI"]).astype(
             float)))).astype(float))) / 2.0) + (data["Ins_Age"] * np.tanh(data["Medical_History_5"]))) / 2.0) +
         np.minimum((0.318310), (np.minimum((((((data["Employment_Info_1"] + data["Medical_Keyword_21"]) / 2.0) + (
         (data["Medical_History_6"] >= np.floor(data["Insurance_History_8"])).astype(float))) / 2.0)), (np.sinh(
             np.floor(np.sinh((data["Insurance_History_5"] * (
             (data["Insurance_History_4"] <= np.tanh(data["Employment_Info_2"])).astype(float)))))))))) +
         np.minimum((0.138462), ((((-(
         np.floor(np.abs((data["Product_Info_2_N"] * data["Product_Info_2_N"]))))) >= np.minimum(
             (data["Product_Info_2_C"]),
             (np.round(np.abs(np.minimum((np.sin(data["Medical_History_2"])), (data["SumMedicalKeywords"]))))))).astype(
             float)))) +
         (-((((math.sin(-1.0) / 2.0) < np.minimum((np.ceil((data["Insurance_History_2"] - ((np.minimum(
             (np.sin(data["Medical_Keyword_40"])), ((-(data["Medical_Keyword_40"])))) < data[
                                                                                                "Medical_History_15"]).astype(
             float))))), (data["Medical_History_15"]))).astype(float)))) +
         np.abs((np.cos(((data["Medical_Keyword_19"] > data["Medical_History_15"]).astype(float))) * (((np.tanh(
             data["Insurance_History_2"]) - ((data["Medical_History_15"] > ((np.cos(
             ((data["Insurance_History_2"] > data["Medical_History_4"]).astype(float))) + data[
                                                                                 "Medical_History_4"]) / 2.0)).astype(
             float))) >= np.round(data["Medical_History_4"])).astype(float)))) +
         (0.058823 * ((data["InsuredInfo_6"] + np.minimum((data["Medical_Keyword_15"]),
                                                          ((data["InsuredInfo_6"] * data["Medical_History_18"])))) + ((
                                                                                                                      data[
                                                                                                                          "Medical_Keyword_28"] + (
                                                                                                                      (
                                                                                                                      -(
                                                                                                                      data[
                                                                                                                          "Medical_History_19"])) + (
                                                                                                                      (
                                                                                                                      data[
                                                                                                                          "Medical_Keyword_3"] *
                                                                                                                      data[
                                                                                                                          "InsuredInfo_6"]) +
                                                                                                                      data[
                                                                                                                          "Medical_Keyword_19"]))) / 2.0))) +
         np.minimum((np.maximum((data["Medical_History_23"]), (data["SumMedicalKeywords"]))), (
         np.minimum(((np.sin(np.maximum((data["Medical_History_17"]), (np.floor(data["SumMedicalKeywords"])))) / 2.0)),
                    (((data["InsuredInfo_7"] >= (
                    (data["Medical_Keyword_38"] * data["SumMedicalKeywords"]) * data["Medical_Keyword_2"])).astype(
                        float)))))) +
         (0.138462 * ((np.cos((np.sinh(data["BMI"]) * 2.0)) + np.cos(((((np.round(data["Medical_History_4"]) < data[
             "Ins_Age"]).astype(float)) - np.sinh(
             ((np.abs(data["Medical_History_22"]) < np.sinh(data["BMI"])).astype(float)))) * 2.0))) / 2.0)) +
         np.minimum(((np.sin(data["Medical_Keyword_32"]) * ((data["Employment_Info_2"] > (
         np.floor(data["Employment_Info_2"]) - (np.minimum((data["Medical_Keyword_37"]), (3.0)) * 2.0))).astype(
             float)))), (np.maximum((np.sin(data["Medical_History_18"])), (data["Medical_History_22"])))) +
         np.sin((data["Medical_History_4"] * np.maximum((data["Medical_Keyword_3"]), (np.sin((data[
                                                                                                  "Medical_History_4"] * np.minimum(
             (0.094340), (((np.maximum((data["Medical_Keyword_34"]), (data["InsuredInfo_5"])) > (
             (-(data["Medical_Keyword_15"])) + data["Medical_History_1"])).astype(float)))))))))) +
         np.sin((np.sin((np.maximum((data["Medical_History_38"]), (np.round((data["Medical_History_35"] + (-(((data[
                                                                                                                   "Medical_Keyword_24"] > np.minimum(
             ((data["Medical_History_7"] * data["Ht"])),
             (np.maximum((data["Employment_Info_2"]), (data["Medical_History_13"]))))).astype(
             float)))))))) * 2.0)) * 2.0)) +
         ((((((np.minimum((data["BMI"]), (data["Insurance_History_2"])) > (
         np.cos(np.cos(data["InsuredInfo_6"])) - np.minimum((data["Ins_Age"]),
                                                            (np.abs(data["InsuredInfo_6"]))))).astype(
             float)) >= np.tanh(((1.584910 + data["SumInsuredInfo"]) / 2.0))).astype(float)) / 2.0) / 2.0) +
         (np.minimum((np.maximum(((np.minimum((data["Medical_History_18"]), (data["Medical_History_10"])) * 2.0)),
                                 ((-(np.minimum((5.200000), (data["BMI"]))))))),
                     (np.minimum((np.tanh(data["SumMedicalKeywords"])), (np.round(data["SumMedicalKeywords"]))))) * (
          ((data["Medical_History_18"] > data["Medical_History_6"]).astype(float)) / 2.0)) +
         np.tanh(((((((((data["InsuredInfo_1"] < data["BMI"]).astype(float)) < (
         (data["InsuredInfo_1"] < data["Medical_History_12"]).astype(float))).astype(float)) < (
                     data["Medical_History_27"] * (data["Medical_History_20"] * data["Medical_History_5"]))).astype(
             float)) * data["BMI"]) / 2.0)) +
         ((((data["Medical_History_9"] > (
         (data["Medical_History_32"] <= (20.750000 + data["InsuredInfo_5"])).astype(float))).astype(float)) >= ((
                                                                                                                np.minimum(
                                                                                                                    (
                                                                                                                    data[
                                                                                                                        "InsuredInfo_5"]),
                                                                                                                    (
                                                                                                                    data[
                                                                                                                        "Medical_Keyword_1"])) <= (
                                                                                                                -(
                                                                                                                np.minimum(
                                                                                                                    (
                                                                                                                    data[
                                                                                                                        "InsuredInfo_5"]),
                                                                                                                    (
                                                                                                                    np.minimum(
                                                                                                                        (
                                                                                                                        data[
                                                                                                                            "InsuredInfo_6"]),
                                                                                                                        (
                                                                                                                        np.cos(
                                                                                                                            data[
                                                                                                                                "Medical_History_40"])))))))).astype(
             float))).astype(float)))
    return p


def Individual6(data):
    p = (((5.200000 + np.sin(
        ((data["Medical_History_23"] > ((data["Medical_Keyword_3"] + data["BMI"]) / 2.0)).astype(float)))) - ((((data[
                                                                                                                     "Medical_Keyword_3"] + np.maximum(
        (data["BMI"]), (data["SumMedicalKeywords"]))) / 2.0) + (data["BMI"] - data["Medical_History_4"])) / 2.0)) +
         ((((data["Medical_History_15"] + np.minimum(((data["Medical_History_40"] - data["Insurance_History_2"])), (
         np.minimum(((data["Product_Info_4"] - data["Medical_History_30"])),
                    (np.ceil((data["Medical_History_13"] - data["InsuredInfo_5"]))))))) / 2.0) + np.tanh(
             (np.ceil(data["Product_Info_4"]) - data["InsuredInfo_7"]))) / 2.0) +
         np.minimum((np.tanh(((data["Medical_History_27"] + np.minimum((data["Medical_History_11"]), ((np.minimum(
             (data["Medical_History_31"]), ((np.round(((data["Medical_History_7"] + data["Product_Info_4"]) / 2.0)) +
                                             data["Medical_History_20"]))) * 2.0)))) / 2.0))),
                    (np.cos(np.maximum((data["Ins_Age"]), (np.sinh(data["Medical_History_5"])))))) +
         (((((data["InsuredInfo_6"] + (
         (data["Medical_History_15"] + data["BMI"]) - data["Medical_History_18"])) / 2.0) + ((((data[
                                                                                                    "Medical_Keyword_25"] -
                                                                                                data[
                                                                                                    "Employment_Info_3"]) +
                                                                                               data[
                                                                                                   "Insurance_History_3"]) + (
                                                                                              data[
                                                                                                  "Product_Info_2_N"] + (
                                                                                              data["Family_Hist_4"] +
                                                                                              data[
                                                                                                  "Medical_Keyword_41"]))) / 2.0)) / 2.0) / 2.0) +
         np.sin(((np.maximum((data["InsuredInfo_2"]), (np.maximum((np.floor(data["Product_Info_2_C"])), (
         ((data["Medical_History_24"] > data["Medical_History_38"]).astype(float)))))) + np.sin(((((data[
                                                                                                        "Medical_History_1"] + (
                                                                                                    (data[
                                                                                                         "Medical_Keyword_33"] +
                                                                                                     data[
                                                                                                         "Medical_History_4"]) / 2.0)) / 2.0) + (
                                                                                                  data[
                                                                                                      "Medical_History_5"] *
                                                                                                  data[
                                                                                                      "Medical_Keyword_3"])) / 2.0))) / 2.0)) +
         np.sin(np.minimum(((data["Medical_History_15"] * (((((data["Medical_History_17"] <= (
         (0.720430 + data["Medical_History_15"]) / 2.0)).astype(float)) * 2.0) - ((data["Product_Info_3"] <= data[
             "Insurance_History_2"]).astype(float))) * 2.0))), ((data["Medical_History_11"] * 2.0)))) +
         (np.cos(8.0) * np.round((((data["Medical_Keyword_38"] + data["Medical_History_35"]) + (
         data["InsuredInfo_1"] - (
         data["Family_Hist_2"] + np.maximum((np.maximum((data["Medical_Keyword_34"]), (data["InsuredInfo_5"]))),
                                            (data["Medical_Keyword_22"]))))) / 2.0))) +
         np.minimum((np.tanh(((1.0 / (1.0 + np.exp(- data["InsuredInfo_6"]))) * (
         (data["InsuredInfo_6"] + np.cos(data["Medical_History_15"])) / 2.0)))), (((np.cos(
             (data["BMI"] - ((data["Medical_History_4"] < (1.0 / (1.0 + np.exp(- data["Ins_Age"])))).astype(float)))) +
                                                                                    data[
                                                                                        "Medical_History_15"]) / 2.0))) +
         ((np.sin(
             np.maximum((data["Medical_History_28"]), ((data["Employment_Info_3"] * data["Medical_History_40"])))) + ((
                                                                                                                      np.maximum(
                                                                                                                          (
                                                                                                                          np.minimum(
                                                                                                                              (
                                                                                                                              data[
                                                                                                                                  "Medical_History_40"]),
                                                                                                                              (
                                                                                                                              np.cos(
                                                                                                                                  (
                                                                                                                                  (
                                                                                                                                  (
                                                                                                                                  (
                                                                                                                                  data[
                                                                                                                                      "InsuredInfo_5"] +
                                                                                                                                  data[
                                                                                                                                      "Medical_History_40"]) / 2.0) +
                                                                                                                                  data[
                                                                                                                                      "Insurance_History_2"]) / 2.0))))),
                                                                                                                          (
                                                                                                                          data[
                                                                                                                              "Medical_History_4"])) <
                                                                                                                      data[
                                                                                                                          "Medical_Keyword_19"]).astype(
             float))) / 2.0) +
         (0.094340 * ((np.minimum(((data["Medical_History_6"] - data["Medical_Keyword_9"])),
                                  (np.minimum((data["Medical_History_39"]), (data["Medical_History_33"])))) + (
                       ((data["SumMedicalKeywords"] * data["Ins_Age"]) - data["Family_Hist_4"]) - (
                       data["Ins_Age"] + data["Product_Info_4"]))) / 2.0)) +
         np.sin(np.sin(np.sin((0.138462 * (((data["Medical_Keyword_45"] + data["Medical_Keyword_15"]) / 2.0) + np.cos((
                                                                                                                      np.ceil(
                                                                                                                          data[
                                                                                                                              "Product_Info_2_N"]) - np.floor(
                                                                                                                          (
                                                                                                                          (
                                                                                                                          data[
                                                                                                                              "Product_Info_2_N"] < np.power(
                                                                                                                              np.abs(
                                                                                                                                  data[
                                                                                                                                      "Medical_History_21"]),
                                                                                                                              data[
                                                                                                                                  "Medical_History_21"])).astype(
                                                                                                                              float)))))))))) +
         np.minimum((((np.tanh(data["Medical_History_16"]) > data["Medical_History_15"]).astype(float))), (np.sin(
             np.round(np.maximum(((data["InsuredInfo_2"] * data["Medical_History_5"])),
                                 ((np.minimum((1.584910), ((-(data["Medical_History_4"])))) * 2.0))))))) +
         (np.minimum((np.minimum(((np.minimum(((1.0 / (1.0 + np.exp(- data["Product_Info_2_C"])))),
                                              (data["Medical_Keyword_32"])) * data["Product_Info_2_C"])),
                                 (np.cos(np.maximum((np.tanh(data["Product_Info_2_C"])), (data["Ins_Age"])))))), (((
                                                                                                                   data[
                                                                                                                       "Employment_Info_1"] + (
                                                                                                                   (
                                                                                                                   data[
                                                                                                                       "Medical_History_33"] >
                                                                                                                   data[
                                                                                                                       "Insurance_History_8"]).astype(
                                                                                                                       float))) / 2.0))) / 2.0) +
         ((((data["Medical_History_40"] - data["Medical_History_30"]) < (
         (np.sin(data["Medical_Keyword_2"]) < data["Medical_Keyword_40"]).astype(float))).astype(float)) * ((data[
                                                                                                                 "Medical_Keyword_3"] >= (
                                                                                                             (((data[
                                                                                                                    "Medical_Keyword_32"] *
                                                                                                                data[
                                                                                                                    "Medical_History_30"]) *
                                                                                                               data[
                                                                                                                   "Medical_Keyword_3"]) +
                                                                                                              data[
                                                                                                                  "Medical_History_11"]) / 2.0)).astype(
             float))) +
         np.minimum(((data["SumMedicalKeywords"] * (
         (data["Medical_History_18"] >= data["Medical_History_39"]).astype(float)))), (np.minimum((0.058823), (
         np.maximum((data["Wt"]), (np.minimum((np.sin(np.ceil(data["Product_Info_4"]))), (
         (data["InsuredInfo_6"] * ((data["Medical_History_19"] > data["Medical_History_39"]).astype(float))))))))))) +
         np.minimum(((0.138462 * ((((data["Medical_History_23"] <= (
         data["Medical_Keyword_25"] - data["Medical_History_13"])).astype(float)) <= (
                                   data["Medical_Keyword_12"] - data["Product_Info_4"])).astype(float)))), (((
                                                                                                             0.094340 - np.minimum(
                                                                                                                 (((
                                                                                                                   data[
                                                                                                                       "InsuredInfo_5"] <
                                                                                                                   data[
                                                                                                                       "Product_Info_4"]).astype(
                                                                                                                     float))),
                                                                                                                 (data[
                                                                                                                      "BMI"]))) / 2.0))) +
         ((np.tanh(((data["Medical_History_23"] >= (
         (data["BMI"] < np.sinh(np.power(np.abs(0.094340), data["Medical_History_32"]))).astype(float))).astype(
             float))) + np.sin((data["Medical_History_13"] * np.floor(
             ((data["Insurance_History_2"] >= data["Medical_History_40"]).astype(float)))))) / 2.0) +
         (((data["Medical_History_22"] < (((np.cos(
             np.minimum((data["Medical_History_22"]), (np.floor(((266) * data["Medical_History_15"]))))) * np.cos(
             data["Medical_Keyword_3"])) < data["Medical_History_19"]).astype(float))).astype(float)) * np.minimum(
             (data["Insurance_History_2"]), ((data["Medical_History_41"] / 2.0)))) +
         np.sinh((np.floor(np.cos(((data["Insurance_History_5"] - 0.367879) + (
         data["Insurance_History_7"] + np.minimum((data["Insurance_History_5"]), (
         (data["Insurance_History_5"] + (data["Insurance_History_4"] - 31.006277)))))))) * 2.0)) +
         (np.sin(((((data["Medical_Keyword_28"] > np.cos(data["Medical_Keyword_6"])).astype(float)) >= np.cos(
             np.maximum((np.minimum((np.round(data["Family_Hist_5"])), (data["BMI"]))), (((data["Medical_Keyword_3"] * (
             (data["InsuredInfo_6"] > data["Medical_History_36"]).astype(float))) + data[
                                                                                              "Medical_Keyword_2"]))))).astype(
             float))) / 2.0) +
         np.minimum((((np.sin(data["Medical_Keyword_41"]) >= data["Medical_History_15"]).astype(float))), (((data[
                                                                                                                 "SumMedicalKeywords"] * (
                                                                                                             (data[
                                                                                                                  "Medical_History_5"] >= np.minimum(
                                                                                                                 (((
                                                                                                                   data[
                                                                                                                       "SumMedicalKeywords"] * (
                                                                                                                   (((
                                                                                                                     data[
                                                                                                                         "InsuredInfo_5"] >=
                                                                                                                     data[
                                                                                                                         "Medical_History_20"]).astype(
                                                                                                                       float)) >=
                                                                                                                    data[
                                                                                                                        "Medical_History_31"]).astype(
                                                                                                                       float))) / 2.0)),
                                                                                                                 (data[
                                                                                                                      "Medical_History_17"]))).astype(
                                                                                                                 float))) / 2.0))) +
         np.tanh((data["Medical_History_4"] * np.sinh((np.minimum((np.minimum(
             (((((-(data["BMI"])) * data["Product_Info_5"]) >= data["Medical_History_17"]).astype(float))),
             (np.maximum((0.138462), (data["Medical_History_15"]))))), (np.maximum(((-(data["BMI"]))), (
         data["Medical_History_15"])))) / 2.0)))) +
         (np.tanh(np.sin(np.tanh(np.sin(data["Medical_Keyword_12"])))) * np.minimum(
             (np.cos(np.floor((-((data["Medical_History_2"] * (997))))))),
             (np.cos((-(((data["Medical_History_2"] * 8.0) * (997)))))))) +
         (0.058823 * np.minimum((np.power(np.abs(data["Product_Info_4"]), (
         (112) * ((data["Product_Info_4"] <= data["Medical_Keyword_15"]).astype(float))))), (((data[
                                                                                                   "Medical_Keyword_15"] * 2.0) + (
                                                                                              np.cos(((1.630430 * data[
                                                                                                  "Employment_Info_2"]) * 2.0)) +
                                                                                              data[
                                                                                                  "Medical_History_29"]))))) +
         ((data["Ins_Age"] * ((((data["Medical_History_5"] >= np.cos(
             np.power(np.abs(data["BMI"]), (np.abs(((data["Ins_Age"] + data["BMI"]) / 2.0)) / 2.0)))).astype(
             float)) >= np.cos(((np.power(np.abs(data["Ins_Age"]), data["Medical_Keyword_23"]) / 2.0) / 2.0))).astype(
             float))) / 2.0) +
         np.sin(np.maximum((data["Medical_Keyword_29"]), (np.maximum((data["Medical_Keyword_25"]), (np.tanh((((data[
                                                                                                                   "Medical_Keyword_8"] *
                                                                                                               data[
                                                                                                                   "Medical_History_18"]) >= np.abs(
             (((np.maximum(((data["Medical_History_15"] / 2.0)), (data["Medical_Keyword_29"])) + data[
                 "Medical_Keyword_29"]) / 2.0) - (data["Medical_History_15"] / 2.0)))).astype(float)))))))) +
         np.tanh(((data["InsuredInfo_5"] * np.maximum(
             (((np.floor(data["Medical_History_40"]) * (1.584910 - data["InsuredInfo_1"])) / 2.0)), (
             np.minimum((data["Employment_Info_3"]), (
             (5.428570 * ((data["Medical_History_28"] > data["Medical_Keyword_38"]).astype(float)))))))) * 2.0)) +
         np.minimum(
             (((data["Medical_History_27"] >= ((data["Medical_History_10"] < (2.44962)).astype(float))).astype(float))),
             (np.sin(np.sin(np.ceil(np.maximum((data["SumMedicalHistory"]), (np.maximum((data["SumEmploymentInfo"]), (
             np.maximum((data["SumMedicalHistory"]), ((data["Medical_History_11"] * data["Ht"])))))))))))))
    return p


def MungeData(train, test):
    le = LabelEncoder()
    train.Product_Info_2.fillna('Z0', inplace=True)
    test.Product_Info_2.fillna('Z0', inplace=True)
    train.insert(0, 'Product_Info_2_N', train.Product_Info_2.str[1:])
    train.insert(0, 'Product_Info_2_C', train.Product_Info_2.str[0])
    train.drop('Product_Info_2', inplace=True, axis=1)
    test.insert(0, 'Product_Info_2_N', test.Product_Info_2.str[1:])
    test.insert(0, 'Product_Info_2_C', test.Product_Info_2.str[0])
    test.drop('Product_Info_2', inplace=True, axis=1)

    le.fit(list(train.Product_Info_2_C) + list(test.Product_Info_2_C))
    train.Product_Info_2_C = le.transform(train.Product_Info_2_C)
    test.Product_Info_2_C = le.transform(test.Product_Info_2_C)

    trainids = train.Id
    testids = test.Id
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    responses = train.Response.values
    train.drop('Response', inplace=True, axis=1)

    train = train.astype(float)
    test = test.astype(float)

    train.insert(0,
                 'SumNAs',
                 len(train.columns) - train.count(axis=1))
    test.insert(0,
                'SumNAs',
                len(test.columns) - test.count(axis=1))
    train.insert(0,
                 'SumMedicalKeywords',
                 train[train.columns[train.columns.str.contains('Medical_Keyword')]]
                 .sum(axis=1, skipna=True))
    test.insert(0,
                'SumMedicalKeywords',
                test[test.columns[test.columns.str.contains('Medical_Keyword')]]
                .sum(axis=1, skipna=True))
    train.insert(0,
                 'SumFamilyHistory',
                 train[train.columns[train.columns.str.contains('Family_Hist')]]
                 .sum(axis=1, skipna=True))
    test.insert(0,
                'SumFamilyHistory',
                test[test.columns[test.columns.str.contains('Family_Hist')]]
                .sum(axis=1, skipna=True))
    train.insert(0,
                 'SumInsuranceHistory',
                 train[train.columns[train.columns.str
                 .contains('Insurance_History')]]
                 .sum(axis=1, skipna=True))
    test.insert(0,
                'SumInsuranceHistory',
                test[test.columns[test.columns.str.contains('Insurance_History')]]
                .sum(axis=1, skipna=True))
    train.insert(0,
                 'SumInsuredInfo',
                 train[train.columns[train.columns.str.contains('InsuredInfo')]]
                 .sum(axis=1, skipna=True))
    test.insert(0,
                'SumInsuredInfo',
                test[test.columns[test.columns.str.contains('InsuredInfo')]]
                .sum(axis=1, skipna=True))
    train.insert(0,
                 'SumEmploymentInfo',
                 train[train.columns[train.columns.str.contains('Employment_Info')]]
                 .sum(axis=1, skipna=True))
    test.insert(0,
                'SumEmploymentInfo',
                test[test.columns[test.columns.str.contains('Employment_Info')]]
                .sum(axis=1, skipna=True))
    train.insert(0,
                 'SumMedicalHistory',
                 train[train.columns[train.columns.str.contains('Medical_History')]]
                 .sum(axis=1, skipna=True))
    test.insert(0,
                'SumMedicalHistory',
                test[test.columns[test.columns.str.contains('Medical_History')]]
                .sum(axis=1, skipna=True))

    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

    train['Response'] = responses
    train.insert(0, 'Id', trainids)
    test.insert(0, 'Id', testids)
    return train, test


if __name__ == "__main__":
    ss = StandardScaler()
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    train, test = MungeData(train, test)
    features = train.columns[1:-1]
    scaledtrain = train.copy()
    scaledtest = test.copy()
    scaledtrain[features] = ss.fit_transform(scaledtrain[features])
    scaledtrain[features] = np.round(scaledtrain[features], 6)
    scaledtest[features] = ss.transform(scaledtest[features])
    scaledtest[features] = np.round(scaledtest[features], 6)
    train[features] = np.round(train[features], 6)
    test[features] = np.round(test[features], 6)
    trainindividual1 = np.clip(Individual1(scaledtrain), 1, 8)
    trainindividual2 = np.clip(Individual2(scaledtrain), 1, 8)
    trainindividual3 = np.clip(Individual3(scaledtrain), 1, 8)
    trainindividual4 = np.clip(Individual4(train), 1, 8)  # Seeing if scaling matters - it doesn't!
    trainindividual5 = np.clip(Individual5(scaledtrain), 1, 8)
    trainindividual6 = np.clip(Individual6(scaledtrain), 1, 8)
    testindividual1 = np.clip(Individual1(scaledtest), 1, 8)
    testindividual2 = np.clip(Individual2(scaledtest), 1, 8)
    testindividual3 = np.clip(Individual3(scaledtest), 1, 8)
    testindividual4 = np.clip(Individual4(test), 1, 8)
    testindividual5 = np.clip(Individual5(scaledtest), 1, 8)
    testindividual6 = np.clip(Individual6(scaledtest), 1, 8)

    trainPredictions = (
                       trainindividual1 + trainindividual2 + trainindividual3 + trainindividual4 + trainindividual5 + trainindividual6) / 6.
    testPredictions = (
                      testindividual1 + testindividual2 + testindividual3 + testindividual4 + testindividual5 + testindividual6) / 6.
    cutoffs = [2.951759, 3.653780, 4.402781, 4.911808, 5.543988, 6.135754, 6.716891]
    trainPredictions = np.digitize(trainPredictions, cutoffs) + 1
    testPredictions = np.digitize(testPredictions, cutoffs) + 1
    pdtrain = pd.DataFrame({"Id": train.Id.astype(int), "Prediction": trainPredictions.astype(int),
                            "Response": train.Response.astype(int)})
    pdtrain = pdtrain.set_index('Id')
    pdtrain.to_csv('gptrain.csv')
    pdtest = pd.DataFrame({"Id": test.Id.astype(int), "Response": testPredictions.astype(int)})
    pdtest = pdtest.set_index('Id')
    pdtest.to_csv('gptest.csv')

