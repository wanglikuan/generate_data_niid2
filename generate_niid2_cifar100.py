from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from tqdm import trange
import numpy as np
import random
import json
import os
import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# data变成train_data
# 把targets变成train_labels

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

for _, train_data in enumerate(trainloader,0):
    trainset.data, trainset.targets = train_data
for _, train_data in enumerate(testloader,0):
    testset.data, testset.targets = train_data

random.seed(1)
np.random.seed(1)
NUM_USERS = 20 # should be muitiple of 10
NUM_LABELS = 10

# numran1 = random.randint(10, 50)
# numran2 = random.randint(1, 10)
# num_samples = (num_samples) * numran2 + numran1 #+ 100

# Setup directory for train/test data
train_path = './data/train/cifar100_niid2_train.json'
test_path = './data/test/cifar100_niid2_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

cifa_data_image = []
cifa_data_label = []

cifa_data_image.extend(trainset.data.cpu().detach().numpy())
cifa_data_image.extend(testset.data.cpu().detach().numpy())
cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
cifa_data_label.extend(testset.targets.cpu().detach().numpy())
cifa_data_image = np.array(cifa_data_image)
cifa_data_label = np.array(cifa_data_label)

cifa_data = []
for i in trange(100):
    idx = cifa_data_label==i
    cifa_data.append(cifa_data_image[idx])


print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
users_lables = []

###### CREATE USER DATA SPLIT #######
# Assign 100 samples to each user
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
idx = np.zeros(100, dtype=np.int64)
# for user in range(NUM_USERS):
#     for j in range(NUM_LABELS):  # 3 labels for each users
#         #l = (2*user+j)%10
#         l = (user + j) % 10
#         print("L:", l)
#         X[user] += cifa_data[l][idx[l]:idx[l]+10].tolist()
#         y[user] += (l*np.ones(10)).tolist()
#         idx[l] += 10

print("IDX1:", idx)  # counting samples for each labels

# Assign remaining sample by power law
user = 0
# props = np.random.lognormal(
#     0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
# props = np.array([[[len(v)-NUM_USERS]] for v in cifa_data]) * \
#     props/np.sum(props, (1, 2), keepdims=True)
# print("here:",props/np.sum(props,(1,2), keepdims=True))
#props = np.array([[[len(v)-100]] for v in mnist_data]) * \
#    props/np.sum(props, (1, 2), keepdims=True)
#idx = 1000*np.ones(10, dtype=np.int64)
# print("here2:",props)


################user_lable_sheet代表每个client的label；要与NUM_LABELS匹配######
user_lable_sheet=[[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],\
    [0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],\
        [0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],\
            [0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]]

################number_samples代表每类采样数量；要与user_lable_sheet的数量匹配###
user_number_samples = [50,50,10,10,10,10,10,10,10,10,\
    10,50,50,10,10,10,10,10,10,10,\
        10,10,50,50,10,10,10,10,10,10,\
            10,10,10,50,50,10,10,10,10,10,\
                10,10,10,10,50,50,10,10,10,10,\

                    10,10,10,10,10,50,50,10,10,10,\
                        10,10,10,10,10,10,50,50,10,10,\
                            10,10,10,10,10,10,10,50,50,10,\
                                10,10,10,10,10,10,10,10,50,50,\
                                    50,10,10,10,10,10,10,10,10,50,\

    50,50,10,10,10,10,10,10,10,10,\
        10,50,50,10,10,10,10,10,10,10,\
            10,10,50,50,10,10,10,10,10,10,\
                10,10,10,50,50,10,10,10,10,10,\
                    10,10,10,10,50,50,10,10,10,10,\

                        10,10,10,10,10,50,50,10,10,10,\
                            10,10,10,10,10,10,50,50,10,10,\
                                10,10,10,10,10,10,10,50,50,10,\
                                    10,10,10,10,10,10,10,10,50,50,\
                                        50,10,10,10,10,10,10,10,10,50]




multi_num = 3
count = 0
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        # l = (2*user+j)%10
        l = user_lable_sheet[user][j]
        num_samples =  user_number_samples[count] # num sample
        count = count + 1
        if(NUM_USERS <= 20): 
            num_samples = num_samples
        l = l*multi_num        
        for plus in range(multi_num):
            if idx[l+plus] + num_samples < len(cifa_data[l+plus]):
                X[user] += cifa_data[l+plus][idx[l+plus]:idx[l+plus]+num_samples].tolist()
                y[user] += ((l+plus)*np.ones(num_samples)).tolist()
                idx[l+plus] += num_samples
                print("check len os user:", user, j, l+plus,
                    "len data", len(X[user]), num_samples)

print("IDX2:", idx) # counting samples for each labels

# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

# Setup 5 users
# for i in trange(5, ncols=120):
for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)

    X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.7, stratify=y[i])

    train_data["user_data"][uname] = {'x': X_train, 'y': y_train}
    train_data['users'].append(uname)
    train_data['num_samples'].append(len(y_train))
    
    test_data['users'].append(uname)
    test_data["user_data"][uname] = {'x': X_test, 'y': y_test}
    test_data['num_samples'].append(len(y_test))

print("Num_samples train :", train_data['num_samples'])
print("Num_samples test :", test_data['num_samples'])
print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")