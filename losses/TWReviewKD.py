import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Translate(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Translate, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,(1,1),(1,1),(0,0),bias=False),
            nn.ReLU(inplace=True),
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

        for in_channel in in_channels:
            translate_teacher_1.append(Translate(in_channel,out_channels[0]))
            translate_teacher_2.append(Translate(in_channel,out_channels[1]))
            translate_teacher_3.append(Translate(in_channel,out_channels[2]))

        self.translate_teacher_1=translate_teacher_1
        self.translate_teacher_2=translate_teacher_2
        self.translate_teacher_3=translate_teacher_3


    def forward(self, features_student, features_teacher):
        # get features
        features_student = features_student[:3]
        features_teacher = features_teacher[:3]

        fkd_student_weight=[self.fkd_matrix(i) for i in features_student]
        fkd_teacher_weight=[self.fkd_matrix(i) for i in features_teacher]

        tmp_teacher_1=[]
        tmp_teacher_2=[]
        tmp_teacher_3=[]

        for feature,module in zip(features_student,self.translate_teacher_1):
            tmp_teacher_1.append(module(feature,features_teacher[0]))
        for feature,module in zip(features_student,self.translate_teacher_2):
            tmp_teacher_2.append(module(feature,features_teacher[1]))
        for feature,module in zip(features_student,self.translate_teacher_3):
            tmp_teacher_3.append(module(feature,features_teacher[2]))

        tmp_teacher_1_weight=self.compute_weight(fkd_teacher_weight[0],
                                                 fkd_student_weight[0],
                                                 fkd_student_weight[1],
                                                 fkd_student_weight[2])

        teacher_1=(torch.stack(tmp_teacher_1,0) * tmp_teacher_1_weight.view(-1,1,1,1,1)).sum(0)
        tmp_teacher_2_weight=self.compute_weight(fkd_teacher_weight[1],
                                                 fkd_student_weight[0],
                                                 fkd_student_weight[1],
                                                 fkd_student_weight[2])

        teacher_2=(torch.stack(tmp_teacher_2,0) * tmp_teacher_2_weight.view(-1,1,1,1,1)).sum(0)
        tmp_teacher_3_weight=self.compute_weight(fkd_teacher_weight[2],
                                                 fkd_student_weight[0],
                                                 fkd_student_weight[1],
                                                 fkd_student_weight[2])

        teacher_3=(torch.stack(tmp_teacher_3,0) * tmp_teacher_3_weight.view(-1,1,1,1,1)).sum(0)
        # losses
        loss = self.hcl_loss([teacher_1,teacher_2,teacher_3],features_teacher)
        return loss

    def compute_weight(self,s,t1,t2,t3,tem=0.07):
        d1=-(s-t1).abs()/tem
        d2=-(s-t2).abs()/tem
        d3=-(s-t3).abs()/tem
        d1=d1.item()
        d2=d2.item()
        d3=d3.item()

        weight=torch.softmax(torch.Tensor([d1,d2,d3]).to(s.device),0)
        return weight

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
            n, c, h, w = fs.shape
            loss = F.mse_loss(fs, ft, reduction="mean")
            cnt = 1.0
            tot = 1.0
            for l in [4, 2, 1]:
                if l >= h:
                    continue
                tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
                tmpft = F.adaptive_avg_pool2d(ft, (l, l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss
        return loss_all