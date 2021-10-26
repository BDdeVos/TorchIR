import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def identity_grid(shape, stackdim, dtype=torch.float32, device="cpu"):
    """Create an nd identity grid."""
    tensors = (torch.arange(s, dtype=dtype, device=device) for s in shape)
    return torch.stack(
        torch.meshgrid(*tensors)[::-1], stackdim
    )  # z,y,x shape and flip for x, y, z coords


class Resampler(nn.Module):
    """
    Generic resampler for 2D and 3D images.
    Expects voxel coordinates as coord_field
    Args:
        input (Tensor): input batch (N x C x IH x IW) or (N x C x ID x IH x IW)
        grid (Tensor): flow-field of size (N x OH x OW x 2) or (N x OD x OH x OW x 3)
        padding_mode (str): padding mode for outside grid values
            'zeros' | 'border'. Default: 'zeros'
    """

    def __init__(
        self, coord_dim: int = 1, mode: str = "bilinear", padding_mode: str = "border"
    ):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.coord_dim = coord_dim

    def forward(self, input, coord_field):
        im_shape = input.shape[2:]
        assert coord_field.shape[self.coord_dim] == len(
            im_shape
        )  # number of coordinates should match image dimension

        coord_field = coord_field.movedim(self.coord_dim, -1)

        # scale for pytorch grid_sample function
        max_extent = (
            torch.tensor(
                im_shape[::-1], dtype=coord_field.dtype, device=coord_field.device
            )
            - 1
        )
        coord_field = 2 * (coord_field / max_extent) - 1
        return F.grid_sample(
            input,
            coord_field,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True,
        )


class StableStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.tensor.detach()
        result = ctx.result.detach()
        e = 1e-6
        assert tensor.numel() > 1
        return (
            (2.0 / (tensor.numel() - 1.0))
            * (grad_output.detach() / (result.detach() * 2 + e))
            * (tensor.detach() - tensor.mean().detach())
        )


stablestd = StableStd.apply


class ScaledTanH(nn.Module):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling

    def forward(self, input):
        return torch.tanh(input) * self.scaling

    def __repr__(self):
        return self.__class__.__name__ + "(" + "scaling = " + str(self.scaling) + ")"


class BiasedTanh(nn.Module):
    def __init__(self, scale_in=1.0, scale_out=1.0, bias=0.0):
        super().__init__()
        self.scale_in = scale_in
        self.scale_out = scale_out
        self.bias = bias

    def forward(self, input):
        return torch.tanh(input * self.scale_in) * self.scale_out + self.bias

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "scale_in="
            + str(self.scale_in)
            + ", "
            + "scale_out="
            + str(self.scale_out)
            + ", "
            + "bias="
            + str(self.bias)
            + ")"
        )


class ScalingAF(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return self.scale_factor ** torch.tanh(input)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "scale_factor="
            + str(self.scale_factor)
            + ")"
        )


class IRDataSet(Dataset):
    """
    Wrapper to convert a dataset into a dataset suitable for image registration experiments.
    """

    def __init__(self, ds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ds = ds

    def __len__(self):
        return len(self.ds) * len(self.ds)

    def __getitem__(self, idx):
        fixed_idx = idx // len(self.ds)
        moving_idx = idx % len(self.ds)
        return {"fixed": self.ds[fixed_idx], "moving": self.ds[moving_idx]}
