import os
import random
import numpy as np
import pickle
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import Tensor, context
from mindspore.train.callback import LossMonitor
from mindspore.train import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from models.HH_fc import SNN_Model_HH
from models.LIF_hh_fc import SNN_Model_LIF_hh
from models.LIF_fc import SNN_Model_LIF
from models.fourLIF_fc import SNN_Model_4LIF

device = context.set_context(device_target="GPU")

def train(data_loader, model, optimizer, criterion):
    model.set_train()
    correct0_train = 0
    correct1_train = 0
    running_loss = 0
    count = 0

    for batch in data_loader.create_dict_iterator():
        X = batch['data']
        ts = batch['label']
        outputs = model(X)
        task_loss = 0
        for i in range(n_tasks):
            task_output_i = outputs[:, i, :]
            task_loss += criterion(task_output_i, ts[:, i])

        optimizer.zero_grad()
        task_loss.backward()
        optimizer.step()
        running_loss += task_loss.asnumpy()
        _, predict0 = ms.ops.ArgMaxWithValue(axis=1)(outputs[:, 0, :])
        _, predict1 = ms.ops.ArgMaxWithValue(axis=1)(outputs[:, 1, :])
        correct0_train += (predict0 == ts[:, 0]).sum().asnumpy()
        correct1_train += (predict1 == ts[:, 1]).sum().asnumpy()
        count += X.shape[0]

    return running_loss, correct0_train / count * 100, correct1_train / count * 100

def val(data_loader, model, criterion, epoch):
    model.set_train(False)
    correct0_train = 0
    correct1_train = 0
    running_loss = 0
    count = 0

    for batch in data_loader.create_dict_iterator():
        X = batch['data']
        ts = batch['label']
        outputs = model(X)
        task_loss = 0
        for i in range(n_tasks):
            task_output_i = outputs[:, i, :]
            task_loss += criterion(task_output_i, ts[:, i])
        running_loss += task_loss.asnumpy()
        _, predict0 = ms.ops.ArgMaxWithValue(axis=1)(outputs[:, 0, :])
        _, predict1 = ms.ops.ArgMaxWithValue(axis=1)(outputs[:, 1, :])
        correct0_train += (predict0 == ts[:, 0]).sum().asnumpy()
        correct1_train += (predict1 == ts[:, 1]).sum().asnumpy()
        count += X.shape[0]

    print('=============Validation Start================')
    print('Epoch:', epoch, ' Loss:', running_loss, ' acc1:', correct0_train / count * 100, ' acc2:', correct1_train / count * 100)
    return correct0_train / count * 100, correct1_train / count * 100

def main(model, optimizer, train_loader, test_loader):
    Epoch = 40
    bestacc1 = 0
    bestacc2 = 0

    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    for e in range(Epoch):
        train_loss, train_acc1, train_acc2 = train(train_loader, model, optimizer, criterion)
        acc1, acc2 = val(test_loader, model, criterion, e)
        if (acc1 > bestacc1 and acc2 > bestacc2) or acc1 + acc2 >= bestacc1 + bestacc2:
            bestacc1 = acc1
            bestacc2 = acc2
    print("best_acc1:", bestacc1, "best_acc2:", bestacc2)
    return bestacc1, bestacc2

#---------------------------------------------------------------

with open('data/multi_fashion_and_mnist.pickle', 'rb') as f:
    trainX, trainLabel, testX, testLabel = pickle.load(f)

trainX = Tensor(trainX.reshape(120000, 1, 36, 36), ms.float32)
trainLabel = Tensor(trainLabel, ms.int32)
testX = Tensor(testX.reshape(20000, 1, 36, 36), ms.float32)
testLabel = Tensor(testLabel, ms.int32)
train_set = ds.NumpySlicesDataset({"data": trainX, "label": trainLabel}, shuffle=True)
test_set = ds.NumpySlicesDataset({"data": testX, "label": testLabel}, shuffle=False)

batch_size = 128
train_loader = train_set.batch(batch_size)
test_loader = test_set.batch(batch_size)

for i in range(20):
    seed = i + 1
    print("seed:", seed)
    seed_value = seed
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    ms.set_seed(seed_value)

    for model_name in ["LIF_fc", "4LIF_fc", "HH_fc", "LIF_hh_fc"]:
        print('model established:', model_name)
        n_tasks = 2
        if model_name == "LIF_fc":
            model = SNN_Model_LIF(n_tasks)
        elif model_name == "4LIF_fc":
            model = SNN_Model_4LIF(n_tasks)
        elif model_name == "HH_fc":
            model = SNN_Model_HH(n_tasks)
        elif model_name == "LIF_hh_fc":
            model = SNN_Model_LIF_hh(n_tasks)
        model.to_float(ms.float32)
        optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-4, weight_decay=1e-5)

        acc1, acc2 = main(model, optimizer, train_loader, test_loader)
        print("model:", model_name, "acc1", acc1, "acc2", acc2)
