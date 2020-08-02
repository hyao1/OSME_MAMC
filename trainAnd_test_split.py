import os

image = []
train = []
test = []
classes = []
root = 'CUB_200_2011/CUB_200_2011/'

with open(root + '/classes.txt') as f:
    for r in f:
        data = r.split()
        classes.append(data[1])

for n in range(200):
    for lists in os.listdir(root+'dataset/train/'+classes[n]):
        train.append(root+'dataset/train/'+classes[n] + lists)
'''
os.mknod('_train.txt')
os.mknod('_test.txt')
with open(root + 'train_test_split.txt') as f:
    for r in f:
        if r == '0':
            train.append(r)
'''
for index, trainImage in enumerate(train):
    print(index)
    print(trainImage)

