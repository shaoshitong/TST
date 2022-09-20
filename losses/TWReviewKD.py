import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.quantization import fuse_conv_bn
class Translate(nn.Module):
    def __init__(self,in_channel,out_channel,mid_channel):
        super(Translate, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,(1,1),(1,1),(0,0),bias=False),
                nn.BatchNorm2d(out_channel)
        )
    def forward(self,x,y):
        x = self.conv1(x)
        b,c,h,w=y.shape
        x = F.interpolate(x,(h,w),mode="nearest")
        return x


class TWReviewKD(nn.Module):
    def __init__(self, shapes, out_shapes, in_channels, out_channels, max_mid_channel):
        super().__init__()
        self.shapes = copy.deepcopy(shapes)
        self.out_shapes = copy.deepcopy(out_shapes)
        self.in_channels = in_channels  # student
        self.out_channels = out_channels  # teacher
        self.max_mid_channel = max_mid_channel

        translate_teacher_1 = nn.ModuleList()
        translate_teacher_2 = nn.ModuleList()
        translate_teacher_3 = nn.ModuleList()

        mid_channel = min(512,out_channels[-1])
        for in_channel in in_channels:
            translate_teacher_1.append(Translate(in_channel,out_channels[0],mid_channel))
            translate_teacher_2.append(Translate(in_channel,out_channels[1],mid_channel))
            translate_teacher_3.append(Translate(in_channel,out_channels[2],mid_channel))


        self.translate_teacher_1=translate_teacher_1
        self.translate_teacher_2=translate_teacher_2
        self.translate_teacher_3=translate_teacher_3

        self.out_embeddings=nn.ModuleList([])
        for out_channel in out_channels:
            self.out_embeddings.append(
                nn.Sequential(nn.Conv2d(out_channel,out_channel,(3,3),(1,1),(1,1),bias=False),
                              nn.BatchNorm2d(out_channel)))

        self.linear1=nn.Linear(4,3)
        self.linear2=nn.Linear(4,3)
        self.linear3=nn.Linear(4,3)

        self.iter_number=0

    def forward(self, features_student, features_teacher):
        # get features
        features_student = [i for i in features_student[:3]]
        features_teacher = [i for i in features_teacher[:3]]


        fkd_teacher_weight=[self.fkd_matrix(i) for i in features_teacher]
        fkd_student_weight=[self.fkd_matrix(i) for i in features_student]

        # print("teacher",fkd_teacher_weight,"student",fkd_student_weight)
        self.iter_number+=1
        tmp_teacher_1=[]
        tmp_teacher_2=[]
        tmp_teacher_3=[]
        for feature,module in zip(features_student,self.translate_teacher_1):
            tmp_teacher_1.append(module(feature,features_teacher[0]))
        for feature,module in zip(features_student,self.translate_teacher_2):
            tmp_teacher_2.append(module(feature,features_teacher[1]))
        for feature,module in zip(features_student,self.translate_teacher_3):
            tmp_teacher_3.append(module(feature,features_teacher[2]))


        tmp_teacher_1_weight=torch.softmax(
            self.linear1(
                torch.stack(
                    [fkd_student_weight[0],fkd_student_weight[1],fkd_student_weight[2],fkd_teacher_weight[0]])),0)

        teacher_1=(torch.stack(tmp_teacher_1,0) * tmp_teacher_1_weight.view(-1,1,1,1,1)).sum(0)

        tmp_teacher_2_weight=torch.softmax(
            self.linear2(
                torch.stack(
                    [fkd_student_weight[0],fkd_student_weight[1],fkd_student_weight[2],fkd_teacher_weight[1]])),0)

        teacher_2=(torch.stack(tmp_teacher_2,0) * tmp_teacher_2_weight.view(-1,1,1,1,1)).sum(0)

        tmp_teacher_3_weight=torch.softmax(
            self.linear3(
                torch.stack(
                    [fkd_student_weight[0],fkd_student_weight[1],fkd_student_weight[2],fkd_teacher_weight[2]])),0)

        teacher_3=(torch.stack(tmp_teacher_3,0) * tmp_teacher_3_weight.view(-1,1,1,1,1)).sum(0)
        # losses
        if random.random()>0.99:
            print(tmp_teacher_1_weight,tmp_teacher_2_weight,tmp_teacher_3_weight)
        student_out_embeddings=[]
        for feature_map,embedding in zip([teacher_1,teacher_2,teacher_3],self.out_embeddings):
            student_out_embeddings.append(embedding(feature_map))
        loss = self.hcl_loss(student_out_embeddings,features_teacher)
        return loss


    def fkd_matrix(self,feature_map):
        feature_map = F.adaptive_avg_pool2d(feature_map,(1,1))
        feature_map = torch.flatten(feature_map,1)
        cosine_result = F.cosine_similarity(feature_map.unsqueeze(0),feature_map.unsqueeze(1),2)
        b1,b2 = cosine_result.shape
        mask = torch.eye(b1).to(cosine_result.device).bool()
        cosine_result = cosine_result[~mask]
        return cosine_result.mean()

    def hcl_loss(self,fstudent, fteacher):
        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            loss = F.mse_loss(fs, ft, reduction="mean")
            loss_all += loss
        return loss_all