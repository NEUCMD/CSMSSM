"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """Convolutional module: Conv1D + BatchNorm + (optional) ReLU."""

    def __init__(
        self,
        n_in_channels: int,
        n_out_channels: int,
        kernel_size: int,
        padding_mode: str = "replicate",
        include_relu: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv1d(
                in_channels=n_in_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding="same",
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(num_features=n_out_channels),
        ]
        if include_relu:
            layers.append(nn.ReLU())
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        return out


def manual_pad(x: torch.Tensor, min_length: int) -> torch.Tensor:
    """
    Manual padding function that pads x to a minimum length with replicate padding.
    PyTorch padding complains if x is too short relative to the desired pad size, hence this function.

    :param x: Input tensor to be padded.
    :param min_length: Length to which the tensor will be padded.
    :return: Padded tensor of length min_length.
    """
    # Calculate amount of padding required
    pad_amount = min_length - x.shape[-1]
    # Split either side
    pad_left = pad_amount // 2
    pad_right = pad_amount - pad_left
    # Create left and right padding tensors by replicating the first and last slices
    pad_left_tensor = x[:, :, 0:1].expand(-1, -1, pad_left)
    pad_right_tensor = x[:, :, -1:].expand(-1, -1, pad_right)
    
    # Concatenate the tensors
    padded_x = torch.cat((pad_left_tensor, x, pad_right_tensor), dim=-1)
    return padded_x