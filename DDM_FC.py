import numpy as np
import torch
import tflearn
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from tensorboardX import SummaryWriter
from progressbar import ProgressBar
from datetime import datetime
import os
from time import time

# import MNIST dataset
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=False)
print(X.shape, testX.shape)

# instantiate the tensorboard summary writers
now = datetime.now()
name = 'DDM_FC_{}/{}/{}_{}:{}'.format(now.month, now.day, now.year, now.hour, now.minute)
writer = SummaryWriter(os.path.join('runs', name))
n_classes = 10
test_batch_size = 32

# just take two classes for now
X, testX = X[Y<n_classes, :], testX[testY<n_classes, ...]
Y, testY = Y[Y<n_classes], testY[testY<n_classes]

# standardize each feature
X_mean, X_std = np.mean(X, 0), np.std(X, 0)
X = (X - X_mean) / (X_std + 1e-10)
testX = (testX - X_mean) / (X_std + 1e-10)


def load_batch(data, labels, mode='train', start_idx=None):
    if mode == 'train':
        rand_sel = np.random.randint(0, X.shape[0], 1)[0]
        x = data[rand_sel, :][None, :]
        y = np.array([labels[rand_sel]])

        x = Variable(torch.from_numpy(x).cuda().float())
        y = Variable(torch.from_numpy(y).cuda().long())

    elif mode == 'test':
        assert(start_idx is not None), 'Please provide start_idx value for test mode.'

        x = data[start_idx, :][None, :]
        y = np.array([labels[start_idx]])

        x = Variable(torch.from_numpy(x).cuda().float())
        y = Variable(torch.from_numpy(y).cuda().long())

    return x, y


class DriftNet(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(DriftNet, self).__init__()
        self.input_dim = input_dim
        self.dt = 0.5
        #self.dt = nn.Parameter(torch.rand(1, requires_grad=True).cuda())j

        self.fc1 = nn.DataParallel(nn.Linear(self.input_dim, 1024).cuda())
        self.fc2 = nn.DataParallel(nn.Linear(1024, 512).cuda())
        self.fc3 = nn.DataParallel(nn.Linear(512, 256).cuda())
        self.fc4 = nn.DataParallel(nn.Linear(256, 128).cuda())
        self.fc5 = nn.DataParallel(nn.Linear(128, n_classes).cuda())

        self.objective = nn.CrossEntropyLoss().cuda()


    def normalize_layer(self, layer):
        return (layer - torch.mean(layer)) / (torch.std(layer) + 1e-12)


    def add_noise(self, layer):
        noise = torch.randn(layer.size(1)).cuda() * (layer * 0.005)
        return layer * self.dt + noise


    def forward(self, input_data):
        max_out = .001
        min_out = .001
        steps = 0
        start_time = time()

        while max_out < 1.1:
            out1 = self.add_noise(self.normalize_layer(self.fc1(input_data)))
            out2 = self.add_noise(self.normalize_layer(self.fc2(out1)))
            out3 = self.add_noise(self.normalize_layer(self.fc3(out2)))
            out4 = self.add_noise(self.normalize_layer(self.fc4(out3)))
            output = self.add_noise(self.normalize_layer(self.fc5(out4)))

            # get the max value of the output layer
            max_out = torch.max(output, dim=1)[0]
            max_out = max_out.data.cpu().numpy()[0]

            # get the min value of the output layer
            # min_out = torch.min(output, dim=1)[0]
            # min_out = min_out.data.cpu().numpy()[0]

            steps += 1

            if steps >= 100:
                break

            #print(output.data.cpu().numpy()[0])
        #print(steps)

        return output, steps, np.round(time() - start_time, 2)


driftnet = DriftNet(784, n_classes)
opt = optim.Adam(driftnet.parameters(), lr=.0005)


for iter in range(100000):
    print(iter)
    x, y = load_batch(X, Y)

    output, _, _ = driftnet(x)  # get model output for batch inputs
    loss = driftnet.objective(output, y)  # calculate loss based on outputs

    # perform a step down the gradient
    opt.zero_grad()
    loss.backward()
    opt.step()

    if iter % 2000 == 0:
        test_outputs = []
        test_labels = []
        val_loss = 0.
        n_steps = 0.
        RT = 0.
        bar = ProgressBar()

        for test_iter in bar(range(testX.shape[0])):
            testx, testy = load_batch(testX, testY, mode='test', start_idx=test_iter)

            test_output, steps, rt = driftnet(testx)
            RT += rt
            n_steps += steps
            val_loss += driftnet.objective(test_output, testy).data.cpu().numpy()

            y_hat = np.argmax(test_output.data.cpu().numpy(), 1)
            test_outputs.append(y_hat)
            test_labels.append(testy.data.cpu().numpy())

        val_acc = np.mean(np.array(test_outputs) == np.array(test_labels))
        val_loss = val_loss / (test_iter + 1)

        writer.add_scalar('Validation Accuracy', val_acc, iter)
        writer.add_scalar('Validation Loss', val_loss, iter)
        writer.add_scalar('Number of Steps', n_steps / (test_iter+1), iter)
        writer.add_scalar('Reaction Time', RT / (test_iter+1), iter)

        # writer.add_scalar('FC1 Noise', driftnet.noise1.clone().cpu().data.numpy(), iter)
        # #writer.add_scalar('DT', driftnet.dt.clone().cpu().data.numpy(), iter)
        #
        # writer.add_scalar('FC2 Noise', driftnet.noise2.clone().cpu().data.numpy(), iter)
        #
        # writer.add_scalar('FC3 Noise', driftnet.noise3.clone().cpu().data.numpy(), iter)
        #
        # writer.add_scalar('FC4 Noise', driftnet.noise4.clone().cpu().data.numpy(), iter)
        #
        # writer.add_scalar('FC5 Noise', driftnet.noise5.clone().cpu().data.numpy(), iter)

        for name, param in driftnet.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), iter)

        #print('Validation Accuracy: {}; Validation Loss: {}'.format(val_acc, val_loss))

torch.save(driftnet.state_dict(), 'driftnet.pt')
