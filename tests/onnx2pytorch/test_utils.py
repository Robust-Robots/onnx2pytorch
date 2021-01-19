import torch
import pytest
import numpy as np
from torch import nn

from onnx2pytorch.helpers import to_onnx
from onnx2pytorch.utils import (
    is_constant,
    get_selection,
    assign_values_to_dim,
    get_activation_value,
)


@pytest.fixture
def inp():
    return torch.rand(10, 10)


def test_is_constant():
    a = torch.tensor([1])
    assert is_constant(a)

    a = torch.tensor(1)
    assert is_constant(a)

    a = torch.tensor([1, 2])
    assert not is_constant(a)


def test_get_selection():
    indices = torch.tensor([1, 2, 5])
    with pytest.raises(AssertionError):
        get_selection(indices, -1)

    assert [indices] == get_selection(indices, 0)
    assert [slice(None), indices] == get_selection(indices, 1)


def test_get_selection_2():
    """Behaviour with python lists is unfortunately not working the same."""
    inp = torch.rand(3, 3, 3)
    indices = torch.tensor(0)

    selection = get_selection(indices, 0)
    assert torch.equal(inp[selection], inp[0])

    selection = get_selection(indices, 1)
    assert torch.equal(inp[selection], inp[:, 0])


@pytest.mark.parametrize(
    "val, dim, inplace", [[torch.zeros(4, 10), 0, False], [torch.zeros(10, 4), 1, True]]
)
def test_assign_values_to_dim(inp, val, dim, inplace):
    indices = torch.tensor([2, 4, 6, 8])

    out = inp.clone()
    if dim == 0:
        out[indices] = val
    elif dim == 1:
        out[:, indices] = val

    res = assign_values_to_dim(inp, val, indices, dim, inplace)
    if inplace:
        assert torch.equal(inp, out)
        assert torch.equal(res, out)
    else:
        # input should not be changed when inplace=False
        assert not torch.equal(inp, out)
        assert torch.equal(res, out)


def test_get_activation_value():
    inp = torch.ones(1, 1, 10, 10).numpy()
    model = nn.Sequential(nn.Conv2d(1, 3, 3), nn.Conv2d(3, 1, 3))
    model[0].weight.data *= 0
    model[0].weight.data += 1
    model.eval()

    onnx_model = to_onnx(model, inp.shape)

    activation_name = onnx_model.graph.node[0].output[0]
    value = get_activation_value(onnx_model, inp, activation_name)
    assert value[0].shape == (1, 3, 8, 8)
    a = value[0].round()
    b = 9 * np.ones((1, 3, 8, 8), dtype=np.float32)
    assert (a == b).all()


def test_get_activation_value_2():
    """Get multiple outputs from onnx model."""
    inp = torch.ones(1, 1, 10, 10).numpy()
    model = nn.Sequential(nn.Conv2d(1, 3, 3), nn.Conv2d(3, 1, 3))
    onnx_model = to_onnx(model, inp.shape)

    activation_names = [x.output[0] for x in onnx_model.graph.node]
    values = get_activation_value(onnx_model, inp, activation_names)
    assert values[0].shape == (1, 3, 8, 8)
    assert values[1].shape == (1, 1, 6, 6)