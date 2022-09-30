import matplotlib.pyplot as plt

font1 = {
    "family": "Times New Roman",
    # 'weight': 'bold',
    "style": "normal",
    "size": 8,
}
plt.rc("font", **font1)
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv("./distance_test.csv")

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax.spines["bottom"].set_visible(True)
ax.spines["left"].set_visible(True)
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.set_facecolor("white")
ax.set_title("", font1)
cls_model1 = data.loc[:, "cls_model1"].tolist()
cls_model2 = data.loc[:, "cls_model2"].tolist()
acc_model1 = data.loc[:, "acc_model1"].tolist()
acc_model2 = data.loc[:, "acc_model2"].tolist()
mse_1_model1 = data.loc[:, "mse_1_model1"].tolist()
mse_1_model2 = data.loc[:, "mse_1_model2"].tolist()
mse_2_model1 = data.loc[:, "mse_2_model1"].tolist()
mse_2_model2 = data.loc[:, "mse_2_model2"].tolist()
mse_3_model1 = data.loc[:, "mse_3_model1"].tolist()
mse_3_model2 = data.loc[:, "mse_3_model2"].tolist()


def mean(list):
    return sum(list) / len(list)


print(mean(acc_model1), mean(acc_model2))


x = np.arange(len(cls_model1))
# ax.plot(x,cls_model1,label="cls_wo_review")
# ax.plot(x,cls_model2,label="cls_w_review")
ax.plot(x, mse_1_model1, label="mse_1_wo_review")
ax.plot(x, mse_1_model2, label="mse_1_w_review")
ax.plot(x, mse_2_model1, label="mse_2_wo_review")
ax.plot(x, mse_2_model2, label="mse_2_w_review")
ax.plot(x, mse_3_model1, label="mse_3_wo_review")
ax.plot(x, mse_3_model2, label="mse_3_w_review")
ax.legend(loc=1)
ax.set_xlabel("batch number")
ax.set_ylabel("Distance")
ax = fig.add_subplot(122)
ax.scatter(x, acc_model1, label="acc_wo_review")
ax.scatter(x, acc_model2, label="acc_w_review")
ax.set_xlabel("batch number")
ax.set_ylabel("Top1 test Accuracy")
ax.legend(loc=1)
plt.savefig("./distance_test.png")
plt.show()
