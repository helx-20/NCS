import mindspore.nn as nn
import mindspore
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.common.initializer import Normal
import numpy as np

v_min, v_max = -1e3, 1e3
batch_size = 20
num_epochs = 101
learning_rate = 5e-4
time_window = 5
thresh = 0.3
lens = 0.5
decay = 0.2
num_classes = 10
context.set_context(device_target="GPU")

class ActFun(nn.Cell):

    def __init__(self):
        super(ActFun, self).__init__()
        self.thresh = thresh
        self.lens = lens

    def construct(self, input):
        return ops.gt(input, self.thresh).astype(mindspore.float32)

act_fun = ActFun()

cfg_fc = [512, 50]

class SNN_Model_LIF(nn.Cell):

    def __init__(self, n_tasks):
        super(SNN_Model_LIF, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Dense(50, 10))

        self.fc1 = nn.Dense(36*36*1, cfg_fc[0], weight_init=Normal(0.02))
        self.fc = nn.Dense(cfg_fc[0], cfg_fc[1], weight_init=Normal(0.02))

    def construct(self, input, win=15):
        batch_size = input.shape[0]
        h1_mem = h1_spike = ops.Zeros()((batch_size, cfg_fc[0]), mindspore.float32)
        h1_sumspike = ops.Zeros()((batch_size, cfg_fc[0]), mindspore.float32)
        for step in range(win):
            x = input.view(batch_size, -1)
            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike = h1_sumspike + h1_spike

        outs = self.fc(h1_sumspike / win)

        output = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            output.append(layer(outs))
        return ops.Stack(axis=1)(output)

def mem_update(fc, x, mem, spike):
    mem = mem * decay * (1 - spike) + fc(x)
    spike = act_fun(mem)
    return mem, spike
