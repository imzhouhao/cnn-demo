import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from datasets import mnist as mnist
import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；3x3 4x4卷积核
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 4)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def getAccuracy(out,labels):
    tot = out.size()[0]
    return torch.sum(torch.eq(torch.argmax(out,dim=1).int(),labels.int())).float()/tot

def criterion(out,labels):
    t = nn.CrossEntropyLoss()
    return t(out,labels.long())

Demo = True
if Demo: train_size = 0;
else: train_size = 60000
test_size = 10000
mnist_data = mnist(train_size,test_size)

learning_rate = 0.01
generation = 1000
batch_size = 100
d = (learning_rate-0.0001)/100.0

net = Net()
print(net)

out = None;in_labels = None;
rand_x = None

if not Demo:
	for i in tqdm.tqdm(range(generation)):
	    rand_x = np.random.randint(0,train_size,batch_size)
	    in_image = mnist_data.train_images[rand_x]
	    in_image = torch.from_numpy(np.expand_dims(in_image,axis=1)).float()
	    in_labels = mnist_data.train_labels[rand_x]
	    in_labels = torch.from_numpy(in_labels).float()

	    out = net(in_image)
	    loss = criterion(out,in_labels)
	    optimizer = optim.SGD(net.parameters(),learning_rate)
	    optimizer.zero_grad()
	    loss.backward()
	    optimizer.step()

	    if i%10==9:
	        test_x = np.random.randint(0,test_size,batch_size)
	        in_image = mnist_data.test_images[test_x]
	        in_image = torch.from_numpy(np.expand_dims(in_image,axis=1)).float()
	        in_labels = mnist_data.test_labels[test_x]
	        in_labels = torch.from_numpy(in_labels).float()

	        out = net(in_image)
	        loss = criterion(out,in_labels)
	        print("Accu:{} Loss:{}".format( getAccuracy(out,in_labels), loss))
	        learning_rate -= d

	## save
	filepath = "./parameters.txt"
	param_dict = net.state_dict()
	torch.save(param_dict,filepath)

##load
filepath = "./parameters.txt"
param_dict = torch.load(filepath)
nnet = Net()
nnet.load_state_dict(param_dict)


## Test && Show
Nrows = 2
Ncols = 3
rand_x = [x for x in range(test_size)]
in_image = mnist_data.test_images[rand_x]
in_image = torch.from_numpy(np.expand_dims(in_image,axis=1)).float()
in_labels = torch.from_numpy(mnist_data.test_labels[rand_x])
out = nnet(in_image)
loss = criterion(out,in_labels)
print("\nTEST:\nAccu:{} Loss:{}".format( getAccuracy(out,in_labels), loss))
while True:
    rand_x = np.random.randint(0,test_size,6)
    for i in range(6):
        plt.subplot(Nrows,Ncols,i+1)
        plt.imshow(mnist_data.test_images[rand_x[i]])
        plt.title( "Label:{} Pred:{}".format(mnist_data.test_labels[rand_x[i]], torch.argmax(out[rand_x[i]]) ) )

    plt.show()
