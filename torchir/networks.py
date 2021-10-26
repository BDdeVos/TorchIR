from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from math import pi
from torch.nn.modules.utils import _ntuple
from .utils import ScaledTanH, ScalingAF
from .transformers import rotation_matrix_eff, scaling_matrix_eff, shearing_matrix_eff


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        downsample=(False,),
        af=nn.ELU,
        ndim=2,
    ):
        Conv = (nn.Conv2d, nn.Conv3d)[ndim - 2]
        AvgPool = (nn.AvgPool2d, nn.AvgPool3d)[ndim - 2]
        padding = kernel_size // 2
        layers = OrderedDict()
        layers["conv"] = Conv(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        if af:
            layers["af"] = af()
        if any(downsample):
            if any(np.asarray(downsample) > 1):  # TODO: improve checking
                layers["downsample"] = AvgPool(tuple(downsample))
        super().__init__(layers)


class DIRNet(nn.Module):
    def __init__(
        self,
        grid_spacing,
        kernel_size=3,
        kernels=32,
        num_conv_layers=5,
        num_dense_layers=2,
        ndim=2,
    ):
        super().__init__()
        grid_spacing = _ntuple(ndim)(grid_spacing)
        self.grid_spacing = grid_spacing

        AF = torch.nn.ELU

        num_downsamplings = np.log2(grid_spacing)
        assert all(
            num_downsamplings == num_downsamplings.round()
        ), "Grid spacing should be factors of 2."
        num_downsamplings = num_downsamplings.astype(int)

        in_channels = 2
        downsample = (0 < num_downsamplings).astype(np.int) + 1  # downsample kernel
        conv_layers = [
            ConvBlock(
                in_channels,
                kernels,
                kernel_size=kernel_size,
                downsample=downsample,
                af=AF,
                ndim=ndim,
            )
        ]
        for i in range(1, num_conv_layers):
            downsample = (i < num_downsamplings).astype(np.int) + 1  # downsample kernel
            conv_layers.append(
                ConvBlock(
                    kernels,
                    kernels,
                    kernel_size=kernel_size,
                    downsample=downsample,
                    af=AF,
                    ndim=ndim,
                )
            )

        dense_layers = [
            ConvBlock(kernels, kernels, kernel_size=1, af=AF, ndim=ndim)
            for _ in range(num_dense_layers)
        ]
        dense_layers.append(
            ConvBlock(kernels, ndim, kernel_size=1, af=None, ndim=ndim)
        )  # linear activation for b-spline coeffs.
        dense_layers[-1].conv.weight.data.fill_(0.0)
        dense_layers[-1].conv.bias.data.fill_(0.0)
        self.conv_layers = nn.Sequential(*conv_layers)
        self.dense_layers = nn.Sequential(*dense_layers)
        self.ndim = ndim

    def forward(self, fixed, moving=None):
        x = fixed
        if moving is not None:
            x = torch.cat((fixed, moving), 1)

        x = self.conv_layers(x)
        x = self.dense_layers(x)

        return x


class AIRNet(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        kernels=32,
        linear_nodes=64,
        num_conv_layers=5,
        num_dense_layers=2,
        num_downsamplings=4,
        ndim=2,
    ):
        super().__init__()
        # TODO: handle anisotropic grid spacings
        assert (
            num_dense_layers >= 1
        ), "Number of dense layers should at least be 1 (excluding the final dense output layer)."

        self.ndim = ndim
        AdaptiveAvgPool = (nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)[ndim - 2]

        AF = nn.ELU

        self.max_scaling = 2
        self.max_rotation = 0.5 * pi
        self.max_shearing = 0.25 * pi

        in_channels = 1
        conv_layers = [
            ConvBlock(
                in_channels,
                kernels,
                kernel_size=kernel_size,
                downsample=(2,) * ndim,
                af=AF,
            )
        ]  # TODO: clean this hacky stuff
        for i in range(1, num_conv_layers):
            downsample = (
                (2,) * ndim if i < num_downsamplings else (1,) * ndim
            )  # TODO: clean this hacky stuff
            conv_layers.append(
                ConvBlock(
                    kernels,
                    kernels,
                    kernel_size=kernel_size,
                    downsample=downsample,
                    af=AF,
                )
            )
        conv_layers.append(AdaptiveAvgPool(1))
        self.convnet_features = nn.Sequential(*conv_layers)

        dense_layers = list()

        dense_layers.append(nn.Linear(2 * kernels, linear_nodes))
        dense_layers.append(AF(inplace=True))
        for i in range(1, num_dense_layers):
            dense_layers.append(nn.Linear(linear_nodes, linear_nodes))
            dense_layers.append(AF(inplace=True))

        self.regression_features = nn.Sequential(*dense_layers)

        self.translation = nn.Linear(linear_nodes, ndim)

        self.rotation = nn.Sequential(
            nn.Linear(linear_nodes, 1 if ndim == 2 else 3),
            ScaledTanH(self.max_rotation),
        )

        self.scaling = nn.Sequential(nn.Linear(linear_nodes, ndim), ScalingAF(2))

        self.shearing = nn.Sequential(
            nn.Linear(linear_nodes, (ndim - 1) * ndim), ScaledTanH(self.max_shearing)
        )

    def forward(self, fixed, moving):
        f = self.convnet_features(fixed)
        m = self.convnet_features(moving)
        x = torch.cat((f.flatten(1), m.flatten(1)), dim=1)
        x = self.regression_features(x)
        translation = self.translation(x)
        rotation = self.rotation(x)
        scale = self.scaling(x)
        shear = self.shearing(x)

        # TODO: The model should probably not be tasked with creating a transformation matrix.
        if self.ndim == 2:
            rot_mat = rotation_matrix_eff(rotation, ndim=self.ndim)
        elif self.ndim == 3:
            rot_mat = (
                rotation_matrix_eff(rotation[:, 0], axis=2, ndim=self.ndim)
                @ rotation_matrix_eff(rotation[:, 1], axis=1, ndim=self.ndim)
                @ rotation_matrix_eff(rotation[:, 2], axis=0, ndim=self.ndim)
            )
        Tmat = (
            scaling_matrix_eff(scale, ndim=self.ndim)
            @ shearing_matrix_eff(shear, ndim=self.ndim)
            @ rot_mat
        )

        return Tmat, translation


class RigidIRNet(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        kernels=32,
        linear_nodes=64,
        num_conv_layers=5,
        num_dense_layers=2,
        num_downsamplings=4,
        ndim=2,
    ):
        super().__init__()
        # TODO: handle anisotropic grid spacings
        self.ndim = ndim
        AdaptiveAvgPool = (nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)[ndim - 2]

        AF = nn.ELU

        self.max_rotation = 0.5 * pi

        in_channels = 1
        conv_layers = [
            ConvBlock(
                in_channels,
                kernels,
                kernel_size=kernel_size,
                downsample=(2,) * ndim,
                af=AF,
            )
        ]  # TODO: clean this hacky stuff
        for i in range(1, num_conv_layers):
            downsample = (
                (2,) * ndim if i < num_downsamplings else (1,) * ndim
            )  # TODO: clean this hacky stuff
            conv_layers.append(
                ConvBlock(
                    kernels,
                    kernels,
                    kernel_size=kernel_size,
                    downsample=downsample,
                    af=AF,
                )
            )
        conv_layers.append(AdaptiveAvgPool(1))
        self.convnet_features = nn.Sequential(*conv_layers)

        dense_layers = list()
        for i in range(0, num_dense_layers):
            dense_layers.append(nn.Linear(linear_nodes, linear_nodes))
            dense_layers.append(AF(inplace=True))

        self.regression_features = nn.Sequential(*dense_layers)

        self.translation = nn.Linear(linear_nodes, ndim)

        self.rotation = nn.Sequential(
            nn.Linear(linear_nodes, 1 if ndim == 2 else 3),
            ScaledTanH(self.max_rotation),
        )

    def forward(self, fixed, moving):
        f = self.convnet_features(fixed)
        m = self.convnet_features(moving)
        x = torch.cat((f.flatten(1), m.flatten(1)), dim=1)
        x = self.regression_features(x)
        translation = self.translation(x)
        rotation = self.rotation(x)

        # TODO: The model should probably not be tasked with creating a transformation matrix.
        if self.ndim == 2:
            Tmat = rotation_matrix_eff(rotation, ndim=self.ndim)
        elif self.ndim == 3:
            Tmat = (
                rotation_matrix_eff(rotation[:, 0], axis=2, ndim=self.ndim)
                @ rotation_matrix_eff(rotation[:, 1], axis=1, ndim=self.ndim)
                @ rotation_matrix_eff(rotation[:, 2], axis=0, ndim=self.ndim)
            )

        return Tmat, translation
