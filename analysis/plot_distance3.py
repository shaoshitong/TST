import matplotlib.pyplot as plt

font1 = {'family':"Times New Roman",
         # 'weight': 'bold',
         'style': 'normal',
         'size': 8,
         }
plt.rc('font',**font1)
import seaborn as sns
import numpy as np

import pandas as pd
data = pd.read_csv("./distance_train.csv")

fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(121)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.set_facecolor('white')
ax.set_title("",font1)
cls_model1 = data.loc[:,'cls_model1'].tolist()
cls_model2 = data.loc[:,'cls_model2'].tolist()
acc_model1 = data.loc[:,'acc_model1'].tolist()
acc_model2 = data.loc[:,'acc_model2'].tolist()
mse_1_model1 = data.loc[:,'mse_1_model1'].tolist()
mse_1_model2 = data.loc[:,'mse_1_model2'].tolist()
mse_2_model1 = data.loc[:,'mse_2_model1'].tolist()
mse_2_model2 = data.loc[:,'mse_2_model2'].tolist()
mse_3_model1 = data.loc[:,'mse_3_model1'].tolist()
mse_3_model2 = data.loc[:,'mse_3_model2'].tolist()

def mean(list):
    return sum(list)/len(list)

print(mean(acc_model1),mean(acc_model2))

def sum_in_patch(list):
    sum_number = 10
    new_list = []
    for i in range(0,len(list),sum_number):
        pp = list[i:i+sum_number]
        new_list.append(mean(pp))
    return new_list

cls_model1 = sum_in_patch(cls_model1)
cls_model2 = sum_in_patch(cls_model2)
acc_model1 = sum_in_patch(acc_model1)
acc_model2 = sum_in_patch(acc_model2)
mse_1_model1 = sum_in_patch(mse_1_model1)
mse_1_model2 = sum_in_patch(mse_1_model2)
mse_2_model1 = sum_in_patch(mse_2_model1)
mse_2_model2 = sum_in_patch(mse_2_model2)
mse_3_model1 = sum_in_patch(mse_3_model1)
mse_3_model2 = sum_in_patch(mse_3_model2)
x = np.arange(len(cls_model1))
# ax.plot(x,cls_model1,label="cls_wo_review")
# ax.plot(x,cls_model2,label="cls_w_review")
ax.plot(x,mse_1_model1,label="mse_1_wo_review")
ax.plot(x,mse_1_model2,label="mse_1_w_review")
ax.plot(x,mse_2_model1,label="mse_2_wo_review")
ax.plot(x,mse_2_model2,label="mse_2_w_review")
ax.plot(x,mse_3_model1,label="mse_3_wo_review")
ax.plot(x,mse_3_model2,label="mse_3_w_review")
ax.legend(loc=1)
ax.set_xlabel("batch number")
ax.set_ylabel("Distance")
ax=fig.add_subplot(122)
ax.scatter(x,acc_model1,label="acc_wo_review")
ax.scatter(x,acc_model2,label="acc_w_review")
ax.set_xlabel("batch number")
ax.set_ylabel("Top1 train Accuracy")
ax.legend(loc=1)
plt.savefig("./distance_train.png")
plt.show()