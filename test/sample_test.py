import torch
import torch.nn as nn
import torch.nn.functional as F
import random,math

A=[1,2,3]
B=1
C=(1,2)
M=[(A,B,C) for i in range(10)]
transitions=random.sample(M,3)
A,B,C=zip(*transitions)
print(A,B,C)