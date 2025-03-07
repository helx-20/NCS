import numpy as np
import mindspore.nn as nn
import mindspore
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

v_min, v_max = -1e3, 1e3
batch_size = 20
num_epochs = 101
learning_rate = 5e-4
time_window = 5
thresh = 0.2
lens = 0.3
device = mindspore.context.set_context(device_target="GPU")

class ActFun(nn.Cell):
    def __init__(self):
        super(ActFun, self).__init__()
        self.gt = P.Greater()
        self.cast = P.Cast()

    def construct(self, input):
        return self.cast(self.gt(input, 0.0), mindspore.float32)

act_fun = ActFun()

cfg_fc = [512, 50]

class HH_neuron(nn.Cell):
    def __init__(self, in_planes, out_planes):
        super(HH_neuron, self).__init__()
        self.fc = nn.Dense(in_planes, out_planes)
        self.fc.weight.set_data(self.fc.weight.data * 0.01)

        self.inp = in_planes
        self.oup = out_planes

        self.V_Na = 115
        self.V_K = -12
        self.V_L = 10.6

        self.gbar_Na = 120
        self.gbar_K = 36
        self.gbar_L = 0.3

        coe_scaling = 1e-2
        self.dt = 1e-2

        learnable = False
        self.a_n_coe = Parameter(coe_scaling * mindspore.ops.StandardNormal()((out_planes,)), requires_grad=learnable)
        self.b_n_coe = Parameter(coe_scaling * mindspore.ops.StandardNormal()((out_planes,)), requires_grad=learnable)
        self.a_m_coe = Parameter(coe_scaling * mindspore.ops.StandardNormal()((out_planes,)), requires_grad=learnable)
        self.b_m_coe = Parameter(coe_scaling * mindspore.ops.StandardNormal()((out_planes,)), requires_grad=learnable)
        self.a_h_coe = Parameter(coe_scaling * mindspore.ops.StandardNormal()((out_planes,)), requires_grad=learnable)
        self.b_h_coe = Parameter(coe_scaling * mindspore.ops.StandardNormal()((out_planes,)), requires_grad=learnable)
    
    def zeros_state(self, size):
        zero_state = [Tensor(np.zeros(size), mindspore.float32)]*4
        return zero_state

    def update_neuron(self, inputs, states=None):
        if states is None:
            v, y, m, h = Tensor(np.zeros_like(inputs.asnumpy()), mindspore.float32), Tensor(np.zeros_like(inputs.asnumpy()), mindspore.float32), Tensor(np.zeros_like(inputs.asnumpy()), mindspore.float32), Tensor(np.zeros_like(inputs.asnumpy()), mindspore.float32)
        else:
            v, y, m, h = states

        a_n = self.a_n_coe
        b_n = self.b_n_coe
        a_m = self.a_m_coe
        b_m = self.b_m_coe
        b_h = self.b_h_coe
        a_h = self.a_h_coe

        g_Na = self.gbar_Na * h * m ** 3
        g_K = self.gbar_K * (y ** 4)
        I_Na = g_Na * (v - self.V_Na)
        I_K = g_K * (v - self.V_K)
        I_L = self.gbar_L * (v - self.V_L)
        
        new_v = v + (inputs - I_Na - I_K - I_L) * self.dt
        new_n = y + (a_n * (1 - y) - b_n * y) * self.dt
        new_m = m + (a_m * (1 - m) - b_m * m) * self.dt
        new_h = h + (a_h * (1 - h) - b_h * h) * self.dt

        spike_out = act_fun(new_v - thresh)
        new_state = (new_v, new_n, new_m, new_h)
        return new_state, spike_out

    def construct(self, input, wins=15):
        batch_size = input.shape[0]
        state1 = self.zeros_state([batch_size, self.oup])
        mems = Tensor(np.zeros([batch_size, wins, self.oup]), mindspore.float32)

        for step in range(wins):
            state1, spike_out = self.update_neuron(self.fc(input[:, step, :]), state1)
            mems[:, step, :] = spike_out
        
        return mems

class SNN_Model_HH(nn.Cell):
    def __init__(self, n_tasks):
        super(SNN_Model_HH, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Dense(50, 10))

        self.fc1 = HH_neuron(36*36*1, cfg_fc[0])
        self.layers = nn.SequentialCell([self.fc1])
        self.fc_output = nn.Dense(cfg_fc[0], cfg_fc[1])

    def construct(self, input, wins=15):
        batch_size = input.shape[0]

        input = input.view(batch_size, -1).astype(mindspore.float32)

        input_seq = P.Stack(axis=1)([input]*wins)
        
        for layer in self.layers:
            input_seq = layer(input_seq)

        output = P.ReduceMean(keep_dims=False)(input_seq, 1)
        outs = self.fc_output(output)

        outputs = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outputs.append(layer(outs))
        return P.Stack(axis=1)(outputs)
