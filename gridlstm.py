import torch
import torch.nn as nn
import torch.nn.modules.rnn as rnn
from torch.autograd import Variable
import unittest
import re

class GridLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, dimensions,
            nonlstm_dimensions=None, bias=True,
            nonlstm_nonlinearity=nn.Tanh, nonlstm_bias=False,
            backwards=False, dilation=1):
        '''
        An implementation of the Grid Long Short-Term Memory
        network described in Kalchbrenner et al. 2016
        (https://arxiv.org/pdf/1507.01526.pdf).
        Rather than having an output dimension, all
        hidden vectors are concatenated and returned.

        Runs in raster order in order of dimensions if backwards is false;
        runs in reverse raster order if backwards is true.

        :param input_channels: the number of input channels
        :type input_channels int:
        :param hidden_channels: the number of hidden channels
        :type hidden_channels: int
        :param dimensions: the number of dimensions for the input
                (not including batch and channel dimensions)
        :type dimensions: int
        :param nonlstm_dimensions: sequence (list) of dimensions
                for which to use recursive fully connected layers rather
                than LSTM layers
        :type nonlstm_dimensions: sequence of ints
        :param bias: Whether to include bias in LSTM dimensions
        :type bias: boolean
        :param nonlstm_nonlinearity: Non-linearity for non-LSTM dimensions
        :type nonlstm_nonlinearity: function or nn.Module subclass
        :param backwards: if True, reads cells in reverse raster order
        :type backwards: boolean
        :param dilation: dilation rate; number of time steps to look back.
                See https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md for a visualization
        :type dilation: int or sequence of ints
        '''
        super().__init__()
        if dimensions <= 0:
            raise ValueError("Dimensions must be >0")
        # Basic settings
        self.dimensions = dimensions
        nonlstm_dimensions = nonlstm_dimensions or set()
        self.hidden_channels = hidden_channels
        self.lstm_dim_map = {}
        self.nonlstm_dim_map = {}
        self.input_channels = input_channels
        self.backwards = backwards
        # Convert dilation to tuple if it isn't already
        if isinstance(dilation, int):
            dilation = (dilation,) * dimensions
        else:
            dilation = tuple(dilation)
            if len(dilation) != dimensions:
                raise ValueError("Dilation is incorrect dimensions.")
        self.dilation = dilation
        # Cells have hidden dims from dimension different direcitons
        #  coming in:
        input_hidden_dims = self.dimensions * hidden_channels
        lstms = []
        nonlstms = []
        for dim in range(self.dimensions):
            if dim in nonlstm_dimensions:
                # Fully connected layer, rather than LSTM cell
                layer = nn.Linear(input_hidden_dims, hidden_channels,
                        bias=nonlstm_bias)
                if nonlstm_nonlinearity is not None:
                    layer = nn.Sequential(layer, nonlstm_nonlinearity())
                nonlstms.append(layer)
                self.nonlstm_dim_map[dim] = len(nonlstms) - 1
            else:
                # LSTM cell in this dimension
                layer = GridLSTMCell(input_channels, input_hidden_dims,
                        hidden_channels, bias=bias)
                lstms.append(layer)
                self.lstm_dim_map[dim] = len(lstms) - 1
        if len(lstms) == 0:
            raise ValueError("No LSTM layers.")
        # Attach modules:
        self.lstms = nn.ModuleList(lstms)
        self.nonlstms = nn.ModuleList(nonlstms)

    def _reset_parameters_test(self):
        # For testing purposes only;
        #  sets all weights to 1 (except forget gates),
        #  sets biases to 0
        for param in self.parameters():
            if len(param.shape) >= 2:
                param.data.fill_(1)
            else:
                param.data.fill_(0)
        for lstm in self.lstms:
            lstm._reset_parameters_test()

    def forward(self, inputs):
        '''
        Run the input through the grid LSTM

        :param inputs: the input tensor (B x D x dimensions...)
        :type inputs: Variable(torch.FloatTensor)
        :returns: the output tensor
                (B x (dimensions * hidden dims) x dimensions...)
        :rtype: Variable(torch.FloatTensor)
        '''
        assert len(inputs.shape) == self.dimensions + 2, \
                ("Expected %d dimensional input," % (self.dimensions + 2)) \
                ("received %d dimensional input." % len(inputs.shape))
        assert inputs.shape[1] == self.input_channels, \
                ("Expected %d channels in input," % self.input_channels) \
                ("received %d channels." % inputs.shape[1])
        # Table to hold intermediate grid states
        # XXX: this could be refactored to be more efficient
        #      (e.g. deallocate state that is no longer needed
        #       when it is used)
        state = self._zero_state(inputs.shape)
        for loc in self._update_list(inputs.shape):
            # Update one grid cell
            self._run_update(loc, inputs, state)
        # Combine state list into output tensor
        output = self._get_output(state)
        return output

    def _get_update_list(self, shape, idx):
        # Recursively yield sequence of update indices
        dim = self.dimensions - 1 - idx
        T = shape[dim + 2]
        for i in range(T):
            if idx + 1 == self.dimensions:
                yield {dim: i}
            else:
                for loc in self._get_update_list(shape, idx + 1):
                    loc[dim] = i
                    yield loc

    def _update_list(self, shape):
        # yield sequence of update indices
        for loc in self._get_update_list(shape, 0):
            yield tuple(val for _, val in sorted(loc.items()))

    def _zero_h(self, N):
        # One cell's zero state (used for h and c)
        return Variable(torch.zeros(N, self.hidden_channels), requires_grad=True)

    def _zero_state(self, shape, idx=0):
        # Recursively create list structure for grid state
        dim = idx
        T = shape[dim + 2]
        extra = 0
        if idx + 1 >= self.dimensions:
            mkstate = lambda: [None for _ in range(self.dimensions)]
            return NdList(mkstate() for _ in range(T + extra))
        else:
            return NdList(self._zero_state(shape, idx + 1)
                    for _ in range(T + extra))

    def _adapt(self, loc):
        # Adapt index depending on whether forwards or backwards
        if self.backwards:
            return tuple(-(i + 1) for i in loc)
        return loc

    def _run_update(self, loc, inputs, state):
        # Run update step for a single grid cell
        # The input for the current grid cell:
        inp = inputs[(slice(None), slice(None)) + self._adapt(loc)]
        # The state for the current grid cell:
        local_state = state[self._adapt(loc)]
        H = []
        C = []
        for dim in range(self.dimensions):
            # Lookback to last location for dimension:
            lloc = list(loc)
            lloc[dim] -= self.dilation[dim]
            if lloc[dim] < 0:
                # Beyond history, use zero state
                N = inputs.shape[0]
                h, c = self._zero_h(N), self._zero_h(N)
            else:
                # Access history
                h, c = state[tuple(self._adapt(lloc))][dim]
            H.append(h)
            C.append(c)
        H = torch.cat(H, 1)
        # Run updates for each dimension
        for dim in range(self.dimensions):
            c = C[dim]
            if dim in self.lstm_dim_map:
                # LSTM update
                cell = self.lstms[self.lstm_dim_map[dim]]
                h, c = cell.forward(inp, (H, c))
            else:
                # Non-LSTM update
                layer = self.nonlstms[self.nonlstm_dim_map[dim]]
                h = layer.forward(H)
            local_state[dim] = (h, c)

    def _get_output(self, state, dim=0):
        # Recursively combine state lists into tensor
        start = 0
        T = state.shape[0]
        end = start + T
        if dim + 1 == self.dimensions:
            # Combine h's at the leaves
            return torch.stack([
                torch.cat([state[i][j][0]
                    for j in range(self.dimensions)], 1)
                for i in range(start, end)], 2)
        else:
            # Combine children at the branches
            return torch.stack([self._get_output(state[i], dim + 1) for i in range(start, end)], 2)

class GridLSTMCell(nn.LSTMCell):
    def __init__(self, input_channels, input_hidden_channels, output_hidden_channels, bias=True):
        rnn.RNNCellBase.__init__(self)
        self.input_size = input_channels
        self.input_hidden_size = input_hidden_channels
        self.hidden_size = output_hidden_channels
        self.weight_ih = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.input_hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * self.hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * self.hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def _reset_parameters_test(self):
        # w_hi, w_hf, w_hc, w_ho
        for param in self.parameters():
            if len(param.shape) == 2:
                hidden_size = param.shape[0] // 4
                param.data.fill_(1)
                # Zero out forget weights
                param.data[hidden_size:hidden_size*2] = 0
            else:
                param.data.fill_(0)

class ExampleBiGridLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, dimensions,
            *args, **kwargs):
        super().__init__()
        kwargs["backwards"] = False
        self.forwards  = GridLSTM(input_channels,
                hidden_channels, dimensions, *args, **kwargs)
        kwargs["backwards"] = True
        self.backwards = GridLSTM(hidden_channels * dimensions,
                hidden_channels, dimensions, *args, **kwargs)

    def forward(self, inputs):
        xf = self.forwards.forward(inputs)
        xb = self.backwards.forward(xf)
        return xf + xb

class NdList:
    def __init__(self, collection=None):
        self.dimensions = None
        self._shape = None
        self.lst = []
        if collection is not None:
            for x in collection:
                self.append(x)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return NdList(self.lst[item])
        elif isinstance(item, tuple):
            assert len(item) <= self.dimensions
            lst = self.lst
            for i, idx in enumerate(item):
                if isinstance(idx, slice):
                    print(item, idx)
                    gen = (lst[j] for j in range(*idx.indices(len(lst))))
                    if i + 1 == len(item):
                        return NdList(gen)
                    else:
                        return NdList(x[tuple(item[i+1:])] for x in gen)
                elif isinstance(idx, int):
                    lst = lst[idx]
                else:
                    raise ValueError("Unsupported index: %s" % type(item))
            return lst
        elif isinstance(item, int):
            return self.lst[item]
        else:
            raise ValueError("Unsupported index: %s" % type(item))

    def __setitem__(self, item, value):
        if isinstance(item, slice):
            idces = list(range(*item.indices(len(self))))
            if not isinstance(value, NdList):
                for i in idces:
                    if self.dimensions == 1:
                        self.lst[i] = value
                    else:
                        self.lst[i][:] = value
            else:
                if value.dimensions != self.dimensions - 1:
                    raise ValueError("Invalid number of dimensions. Expected %d. Got %d." % (self.dimensions - 1, value.dimensions))
                expected_shape = (len(idces),) + self._shape
                if expected_shape != value.shape:
                    raise ValueError("Incompatible shape. Expected %s. Got %s." % (expected_shape, value.shape))
                for i, idx in enumerate(idces):
                    self.lst[idx] = value[i]
        elif isinstance(item, tuple):
            assert len(item) <= self.dimensions and len(item) > 0
            last_lst = None
            lst = self.lst
            for i, idx in enumerate(item):
                if isinstance(idx, slice):
                    raise NotImplementedError("Deep slice updates not yet supported.")
                elif isinstance(idx, int):
                    last_lst = lst
                    lst = lst[idx]
                else:
                    raise ValueError("Unsupported index: %s" % type(item))
            if last_lst is None:
                raise ValueError("Invalid index.", item)
            last_lst[idx] = value
        elif isinstance(item, int):
            if not ((isinstance(value, NdList) and value.dimensions + 1 == self.dimensions) \
                    or (not isinstance(value, NdList) and self.dimensions == 1)):
                raise ValueError("Incompatible shapes.")
            self.lst[item] = value 
        else:
            raise ValueError("Unsupported index: %s" % type(item))

    def __len__(self):
        return len(self.lst)

    def __iter__(self):
        for lst in self.lst:
            yield lst

    def __str__(self):
        return str(self.lst)

    def __repr__(self):
        return repr(self.lst)

    def append(self, x):
        if isinstance(x, NdList):
            if self.dimensions is None:
                self.dimensions = x.dimensions + 1
                self._shape = x.shape
            elif self.dimensions != x.dimensions + 1:
                raise ValueError("Inconsistant number of dimensions")
            elif self._shape != x.shape:
                raise ValueError("Inconsistant shapes")
        else:
            if self.dimensions is None:
                self.dimensions = 1
                self._shape = None
            elif self.dimensions != 1:
                raise ValueError("Inconsistant number of dimensions")
        self.lst.append(x)

    @property
    def shape(self):
        if self.dimensions == 1:
            return (len(self.lst),)
        elif self._shape is None:
            return None
        else:
            return (len(self.lst),) + tuple(self._shape)

def _str2float(expected):
    return [[float(y) for y in re.split(r"[ \t]+", x.strip())]
            for x in expected.strip().split("\n")]

class TestGridLSTM(unittest.TestCase):
    def test_constant_checksum_forward(self):
        lstm = GridLSTM(256, 16, 2)
        lstm._reset_parameters_test()
        x = Variable(torch.ones(2, 256, 5, 3))
        out = lstm.forward(x).data
        self.assertEqual(tuple(out.shape), (2, 32, 5, 3))
        expected = r"""
         48.7420  53.3357  54.4950
          53.3358  57.9295  59.0888
           54.4951  59.0887  60.2480
            54.9005  59.4942  60.6535
             55.0698  59.6635  60.8228"""
        expected = torch.FloatTensor(_str2float(expected))
        checksum = torch.sum(torch.sum(torch.abs(out), 0), 0)
        diff = torch.norm(expected - checksum, p=2)
        self.assertTrue(diff < 1e-2)

    def test_constant_checksum_backward(self):
        lstm = GridLSTM(256, 16, 2, backwards=True)
        lstm._reset_parameters_test()
        x = Variable(torch.ones(2, 256, 5, 3))
        out = lstm.forward(x).data
        self.assertEqual(tuple(out.shape), (2, 32, 5, 3))
        expected = r"""
         60.8228  59.6635  55.0698
          60.6535  59.4942  54.9005
           60.2480  59.0887  54.4951
            59.0888  57.9295  53.3358
             54.4950  53.3357  48.7420
        """
        expected = torch.FloatTensor(_str2float(expected))
        checksum = torch.sum(torch.sum(torch.abs(out), 0), 0)
        diff = torch.norm(expected - checksum, p=2)
        self.assertTrue(diff < 1e-2)

    def test_constant_checksum_forward_middle(self):
        lstm = GridLSTM(256, 16, 2)
        lstm._reset_parameters_test()
        x = Variable(torch.zeros(2, 256, 5, 3))
        x.data[:,:,2,1] = 1
        out = lstm.forward(x).data
        self.assertEqual(tuple(out.shape), (2, 32, 5, 3))
        expected = r"""
          0.0000   0.0000   0.0000
          0.0000   0.0000   0.0000
          0.0000  48.7420  53.3354
          0.0000  53.3354  57.9294
          0.0000  54.4950  59.0887
        """
        expected = torch.FloatTensor(_str2float(expected))
        checksum = torch.sum(torch.sum(torch.abs(out), 0), 0)
        diff = torch.norm(expected - checksum, p=2)
        self.assertTrue(diff < 1e-2)

    def test_constant_checksum_backward_middle(self):
        lstm = GridLSTM(256, 16, 2, backwards=True)
        lstm._reset_parameters_test()
        x = Variable(torch.zeros(2, 256, 5, 3))
        x.data[:,:,2,1] = 1
        out = lstm.forward(x).data
        self.assertEqual(tuple(out.shape), (2, 32, 5, 3))
        expected = r"""
            59.0887  54.4950   0.0000
            57.9294  53.3354   0.0000
            53.3354  48.7420   0.0000
            0.0000   0.0000   0.0000
            0.0000   0.0000   0.0000
        """
        expected = torch.FloatTensor(_str2float(expected))
        checksum = torch.sum(torch.sum(torch.abs(out), 0), 0)
        diff = torch.norm(expected - checksum, p=2)
        self.assertTrue(diff < 1e-2)

    def test_range_checksum_forward_middle(self):
        lstm = GridLSTM(1, 16, 2)
        lstm._reset_parameters_test()
        x = Variable(torch.arange(15).view(1, 1, 5, 3))
        out = lstm.forward(x).data
        self.assertEqual(tuple(out.shape), (1, 32, 5, 3))
        expected = r"""
            0.0000  11.8274  25.8689
            22.5208  28.1021  29.5198
            26.5907  29.3201  30.1239
            27.2235  29.6631  30.3267
            27.4408  29.7955  30.4113
        """
        expected = torch.FloatTensor(_str2float(expected))
        checksum = torch.sum(torch.sum(torch.abs(out), 0), 0)
        diff = torch.norm(expected - checksum, p=2)
        self.assertTrue(diff < 1e-2)

    def test_range_checksum_backward_middle(self):
        lstm = GridLSTM(1, 16, 2, backwards=True)
        lstm._reset_parameters_test()
        x = Variable(torch.arange(15).view(1, 1, 5, 3))
        out = lstm.forward(x).data
        self.assertEqual(tuple(out.shape), (1, 32, 5, 3))
        expected = r"""
            30.4114  29.8317  27.5349
            30.3267  29.7471  27.4502
            30.1240  29.5444  27.2475
            29.5444  28.9647  26.6679
            27.2475  26.6679  24.3710
        """
        expected = torch.FloatTensor(_str2float(expected))
        checksum = torch.sum(torch.sum(torch.abs(out), 0), 0)
        diff = torch.norm(expected - checksum, p=2)
        self.assertTrue(diff < 1e-2)

    def test_range_checksum_nonlstm(self):
        lstm = GridLSTM(1, 16, 2, nonlstm_dimensions=[0])
        lstm._reset_parameters_test()
        self.assertEqual(len(lstm.lstms), 1)
        self.assertEqual(len(lstm.nonlstms), 1)
        x = Variable(torch.arange(15).view(1, 1, 5, 3))
        out = lstm.forward(x).data
        self.assertEqual(tuple(out.shape), (1, 32, 5, 3))
        expected = r"""
              0.0000   5.9137  29.6901
             11.2604  30.4052  31.0380
             12.1387  30.4788  31.0609
             12.1832  30.4822  31.0620
             12.1854  30.4824  31.0620
        """
        expected = torch.FloatTensor(_str2float(expected))
        checksum = torch.sum(torch.sum(torch.abs(out), 0), 0)
        diff = torch.norm(expected - checksum, p=2)
        self.assertTrue(diff < 1e-2)

    def test_range_checksum_nonlstm2(self):
        with self.assertRaises(ValueError):
            GridLSTM(1, 16, 2, nonlstm_dimensions=[0, 1])

    def test_range_checksum_nonlstm(self):
        lstm = GridLSTM(1, 16, 2, nonlstm_dimensions=[1])
        lstm._reset_parameters_test()
        self.assertEqual(len(lstm.lstms), 1)
        self.assertEqual(len(lstm.nonlstms), 1)
        x = Variable(torch.arange(15).view(1, 1, 5, 3))
        out = lstm.forward(x).data
        self.assertEqual(tuple(out.shape), (1, 32, 5, 3))
        expected = r"""
            0.0000   5.9137   9.7325
            11.2604  29.6957  30.2487
            30.4052  30.8377  30.9908
            31.0380  31.1808  31.2371
            31.2553  31.3131  31.3372
        """
        expected = torch.FloatTensor(_str2float(expected))
        checksum = torch.sum(torch.sum(torch.abs(out), 0), 0)
        diff = torch.norm(expected - checksum, p=2)
        self.assertTrue(diff < 1e-2)

if __name__ == "__main__":
    unittest.main()
