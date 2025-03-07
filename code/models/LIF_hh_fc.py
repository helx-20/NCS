import numpy as np
import mindspore.nn as nn
import mindspore
from mindspore import Tensor, ops
from mindspore.ops import functional as F

v_min, v_max = -1e3, 1e3
batch_size = 20
# tau_w  = 30
num_epochs = 101
thresh = 2
lens = 0.4
decay = 0.5
num_classes = 10
device = mindspore.context.set_context(device_target="GPU")

class ActFun(nn.Cell):

    def __init__(self):
        super(ActFun, self).__init__()

    def construct(self, input):
        return ops.gt(input, thresh).astype(mindspore.float32)

act_fun = ActFun()

cfg_fc = [512, 50]

class lif_hh(nn.Cell):
    def __init__(self, in_planes, out_planes):
        super(lif_hh, self).__init__()
        self.fc1 = nn.Dense(in_planes, out_planes)
        self.fc2 = nn.Dense(in_planes, out_planes)
        self.fc3 = nn.Dense(in_planes, out_planes)
        self.lif_fc = nn.Dense(3, 1)
        self.lif_fc.weight.set_data(abs(self.lif_fc.weight.data))
    
    def construct(self, input, mem, spike):
        input1 = self.fc1(input)
        input2 = self.fc2(input)
        input3 = self.fc3(input)
        inner_input = self.lif_fc(mem[:, :, 0:3])
        mem1 = ops.zeros_like(mem)
        spike1 = ops.zeros_like(spike)
        mem1[:, :, 0], spike1[:, :, 0] = mem_update(input1, mem[:, :, 0], spike[:, :, 0])
        mem1[:, :, 1], spike1[:, :, 1] = mem_update(input2, mem[:, :, 1], spike[:, :, 1])
        mem1[:, :, 2], spike1[:, :, 2] = mem_update(input3, mem[:, :, 2], spike[:, :, 2])
        mem1[:, :, 3], spike1[:, :, 3] = mem_update(inner_input[:, :, 0], mem[:, :, 3], spike[:, :, 3])
        return mem1, spike1

class SNN_Model_LIF_hh(nn.Cell):

    def __init__(self, n_tasks):
        super(SNN_Model_LIF_hh, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Dense(50, 10))
        
        self.fc_output = nn.Dense(cfg_fc[0] * 4, cfg_fc[1])
        self.lif_4 = lif_hh(36 * 36 * 1, cfg_fc[0])

    def construct(self, input, win=15):
        batch_size = input.shape[0]
        h1_mem = h1_spike = ops.zeros((batch_size, cfg_fc[0], 4), mindspore.float32)
        h1_sumspike = ops.zeros((batch_size, cfg_fc[0], 4), mindspore.float32)
        for step in range(win):
            x = input.view(batch_size, -1)
            h1_mem, h1_spike = self.lif_4(x, h1_mem, h1_spike)
            h1_sumspike = h1_sumspike + h1_spike
        x = h1_sumspike
        x = x.view(batch_size, -1)
        outs = self.fc_output(x / win)

        output = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            output.append(layer(outs))
        return ops.stack(output, axis=1)

def mem_update(x, mem, spike):
    mem = mem * decay * (1 - spike) + x
    spike1 = act_fun(mem)
    return mem, spike1
