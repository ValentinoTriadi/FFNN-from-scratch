import numpy as np
import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.activationFunction import ActivationFunction

activation_function = ActivationFunction([])
data_test = np.random.rand(3, 3)
torch_data_test = torch.tensor(data_test, requires_grad=False, dtype=torch.float32)


@torch.no_grad()
def test_linear():
    check = activation_function.linear(data_test)
    weight = torch.eye(3)
    bias = torch.zeros(3)
    torch_check = torch.nn.functional.linear(torch_data_test, weight, bias)

    assert np.allclose(check, torch_check.detach().numpy())


@torch.no_grad()
def test_relu():
    check = activation_function.relu(data_test)
    torch_check = torch.nn.functional.relu(torch_data_test)

    assert np.allclose(check, torch_check.detach().numpy())

@torch.no_grad()
def test_sigmoid():
    check = activation_function.sigmoid(data_test)
    torch_check = torch.nn.functional.sigmoid(torch_data_test)

    assert np.allclose(check, torch_check.detach().numpy())

@torch.no_grad()
def test_tanh():
    check = activation_function.tanh(data_test)
    torch_check = torch.nn.functional.tanh(torch_data_test)

    assert np.allclose(check, torch_check.detach().numpy())

# @torch.no_grad()
# def test_softmax():
#     check = activation_function.softmax(data_test)
#     torch_check = torch.nn.functional.softmax(torch_data_test, dim=0)

#     assert np.allclose(check, torch_check.detach().numpy())

if __name__ == "__main__":
    test_linear()
    test_relu()
    test_sigmoid()
    test_tanh()
    # test_softmax()