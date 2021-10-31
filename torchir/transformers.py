import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchir.utils import identity_grid, Resampler


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()


def bspline_kernel_nd(t, order, dtype=float, **kwargs):
    tpowers = t ** torch.arange(order, 0 - 1, -1, dtype=dtype)
    if order == 1:
        return tpowers @ torch.tensor(((-1, 1), (1, 0)), dtype=dtype)
    elif order == 2:
        return (
            tpowers
            @ torch.tensor(((1, -2, 1), (-2, 2, 0), (1, 1, 0)), dtype=dtype)
            / 2.0
        )
    elif order == 3:
        return (
            tpowers
            @ torch.tensor(
                ((-1, 3, -3, 1), (3, -6, 3, 0), (-3, 0, 3, 0), (1, 4, 1, 0)),
                dtype=dtype,
            )
            / 6.0
        )


def bspline_convolution_kernel(upsampling_factors, order, dtype=float):
    ndim = len(upsampling_factors)
    for i, us_factor in enumerate(upsampling_factors):
        t = torch.linspace(1 - (1 / us_factor), 0, us_factor)
        ker1D = bspline_kernel_nd(t[:, None], order, dtype).T.flatten()
        shape = (1,) * i + ker1D.shape + (1,) * (ndim - 1 - i)
        try:
            kernel = kernel * ker1D.view(shape)
        except NameError:
            kernel = ker1D.view(shape)
    return kernel


class BsplineTransformer(Transformer):
    """
    third order bspline. upsampling via transposed convolutions.
    """

    def __init__(
        self,
        network,
        order=3,
        trainable=False,
        return_field=False,
        apply_diffeomorphic_limits=False,
    ):
        super().__init__()
        self.network = network
        self.ndim = network.ndim
        self.upsampling_factors = network.grid_spacing
        self.return_field = return_field
        ConvTranspose = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)[
            self.ndim - 1
        ]

        bspline_kernel = self.make_bspline_kernel(self.upsampling_factors, order=order)
        kernel_size = bspline_kernel.shape
        crop_size = tuple(int(el * 3 / 8) if el != 5 else 2 for el in kernel_size)
        self._upsampler = ConvTranspose(
            1,
            1,
            kernel_size,
            stride=self.upsampling_factors,
            padding=crop_size,
            bias=False,
        )
        self._upsampler.weight = nn.Parameter(
            bspline_kernel[None, None], requires_grad=trainable
        )

        self._resampler = Resampler(coord_dim=1)

        self.apply_diffeomorphic_limits = apply_diffeomorphic_limits

    @staticmethod
    def make_bspline_kernel(upsampling_factors, order, dtype=torch.float32):
        # TODO: solve the issues with kernel shapes
        bspline_kernel = bspline_convolution_kernel(
            upsampling_factors, order=order, dtype=dtype
        )

        if (np.array(bspline_kernel.shape[::-1]) == 4).any() or (
            np.array(bspline_kernel.shape[::-1]) == 2
        ).any():  # hack to deal with 1 strides and kernel size of 4
            padding = list()
            for s in bspline_kernel.shape[::-1]:
                if s == 4 or s == 2:
                    padding.extend([1, 0])
                else:
                    padding.extend([0, 0])
            bspline_kernel = F.pad(bspline_kernel, padding, mode="constant")

        return bspline_kernel

    def dvf(self, bspline_parameters, output_shape):
        assert bspline_parameters.shape[1] == self.ndim
        shape = bspline_parameters.shape
        dvf = self._upsampler(
            bspline_parameters.view((shape[0] * self.ndim, 1) + shape[2:]),
            output_size=output_shape,
        )
        newshape = dvf.shape

        return dvf.view((shape[0], self.ndim) + newshape[2:])

    def forward(self, fixed_image, moving_image, coord_field=None):
        """
        :param fixed_image:
        :param moving_image:
        :param other: Can be image or coordinate field with coords at dimension 1.
        :return:
        """
        bspline_parameters = self.network(fixed_image, moving_image)

        if self.apply_diffeomorphic_limits:
            bspline_parameters = torch.tanh(bspline_parameters) * (
                0.4
                * torch.tensor(
                    self.upsampling_factors[::-1],
                    dtype=bspline_parameters.dtype,
                    device=bspline_parameters.device,
                ).view((1, self.ndim) + (1,) * self.ndim)
            )

        dvf = self.dvf(bspline_parameters, fixed_image.shape[2:])

        grid = dvf + identity_grid(
            dvf.shape[2:], stackdim=0, dtype=dvf.dtype, device=dvf.device
        )

        if coord_field is not None:
            grid = self._resampler(coord_field.movedim(-1, 1), grid)

        if self.return_field:
            warped = grid.movedim(1, -1)
        else:
            warped = self._resampler(moving_image, grid)

        return warped


def scaling_matrix_eff(x, ndim=2):
    assert (
        x.ndim == 2
    ), "Input should be a tensor of (m, n), where m is number of instances, and n number of parameters."
    assert ndim in (2, 3), "Only 2D and 3D implemented."
    assert ndim == x.shape[-1], f"Number of parameters for {ndim}D should be {ndim}."
    dtype, device = x.dtype, x.device
    T = torch.zeros((len(x), ndim, ndim), dtype=dtype, device=device)
    sel_mask = torch.eye(ndim, device=device, dtype=torch.bool)
    T[:, sel_mask] = x
    return T


def rotation_matrix_eff(x, axis=0, ndim=2):
    """
    For 3D axis = x: 2, y: 1, z: 0.
    """
    assert (
        x.ndim == 2 and x.shape[-1] == 1
    ), "Input should be a tensor of (m, 1), where m is number of instances."
    assert ndim in (2, 3), "Only 2D and 3D implemented."
    dtype, device = x.dtype, x.device
    T = torch.eye(ndim, dtype=dtype, device=device)[None].repeat(len(x), 1, 1)
    lidx, hidx = ((0, 1), (0, 2), (1, 2))[axis]
    c = torch.cos(x)
    s = torch.sin(x)
    T[:, lidx, lidx] = c.squeeze()
    T[:, lidx, hidx] = s.squeeze()
    T[:, hidx, lidx] = -s.squeeze()
    T[:, hidx, hidx] = c.squeeze()
    return T


def shearing_matrix_eff(x, ndim=2):
    assert (
        x.ndim == 2
    ), "Input should be a tensor of (m, n), where m is number of instances, and n number of parameters."
    assert ndim in (2, 3), "Only 2D and 3D implemented."
    assert (ndim - 1) * ndim == x.shape[
        -1
    ], f"Number of parameters for {ndim}D should be {(ndim - 1) * ndim}"
    dtype, device = x.dtype, x.device
    T = torch.eye(ndim, dtype=dtype, device=device)[None].repeat(len(x), 1, 1)
    T[:, ~torch.eye(ndim, device=device, dtype=torch.bool)] = torch.tan(x)
    return T


def translation_matrix(x, ndim=2):  # not used for efficient transforms
    assert (
        x.ndim == 2
    ), "Input should be a tensor of (m, n), where m is number of instances, and n number of parameters."
    assert ndim in (2, 3), "Only 2D and 3D implemented."
    assert ndim == x.shape[-1], f"Number of parameters for {ndim}D should be {ndim}."
    dtype, device = x.dtype, x.device
    T = torch.eye(ndim + 1, dtype=dtype, device=device)[None].repeat(len(x), 1, 1)
    T[:, :ndim, ndim] = x
    return T


def tmat_2d_affine(rotation, scale, shear):
    c = torch.cos(rotation[:, 0])
    s = torch.sin(rotation[:, 0])
    sx = scale[:, 0]
    sy = scale[:, 1]
    sh = torch.tan(shear[:, 0])
    T = torch.stack((sx * (c - s * sh), -s * sy, (sx * (s + c * sh)), sy * c)).view(
        -1, 2, 2
    )
    return T


shearmat2d = shearing_matrix_eff
rotmat2d = rotation_matrix_eff
scalemat2d = scaling_matrix_eff


def batch_transform_efficient(coord_field, Tmat, translation, Forigin, Morigin):
    shape = coord_field.shape
    coord_field = coord_field.view(shape[0], -1, shape[-1])  # flatten
    coord_field = coord_field + Forigin[:, None]
    coord_field = coord_field @ Tmat.transpose(
        2, 1
    )  # transpose because of switched order
    coord_field = coord_field + (translation[:, None] - Morigin[:, None])
    shape = (len(coord_field),) + shape[1:]  # enable broadcasting
    coord_field = coord_field.view(shape)  # recover original shape
    return coord_field


def batch_transform(coord_field, Tmat, Forigin, Morigin):
    ndim = coord_field.shape[-1]
    Tmat = translation_matrix(-Morigin) @ Tmat @ translation_matrix(Forigin)
    coord_field = torch.cat(
        (
            coord_field,
            torch.ones(
                coord_field.shape[:-1] + (1,),
                dtype=coord_field.dtype,
                device=coord_field.device,
            ),
        ),
        dim=ndim + 1,
    )  # add extra dimension for MM
    shape = coord_field.shape
    coord_field = coord_field.view(shape[0], -1, shape[-1])  # flatten
    coord_field = coord_field @ Tmat.transpose(2, 1)  # MM
    coord_field = coord_field.view(shape)  # recover shape
    coord_field = coord_field.narrow(ndim + 1, 0, ndim)  # remove extra dimension
    return coord_field


class GlobalTransformer(Transformer):
    def __init__(self, network, return_field=False):
        super().__init__()
        self.network = network  # TODO: probably should do some checking here
        self.ndim = network.ndim
        self._resampler = Resampler(coord_dim=-1)
        self.return_field = return_field

    def forward(self, fixed_image, moving_image, coord_field=None):
        Tmat, translation = self.network(fixed_image, moving_image)
        f_origin = (
            torch.tensor(
                fixed_image.shape[2:],
                dtype=fixed_image.dtype,
                device=fixed_image.device,
            )[None]
            / 2
        )
        m_origin = (
            torch.tensor(
                moving_image.shape[2:],
                dtype=moving_image.dtype,
                device=moving_image.device,
            )[None]
            / 2
        )

        if coord_field is not None:
            assert (
                coord_field.shape[-1] == self.ndim
            ), f"Shape[-1] of coord_field should be {self.ndim}."
        else:
            coord_field = identity_grid(
                fixed_image.shape[2:],
                stackdim=-1,
                dtype=fixed_image.dtype,
                device=fixed_image.device,
            )[None]

        coord_field = batch_transform_efficient(
            coord_field, Tmat, translation, f_origin, m_origin
        )

        if self.return_field:
            warped = coord_field
        else:
            warped = self._resampler(moving_image, coord_field)

        return warped
