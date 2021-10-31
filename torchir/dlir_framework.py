from torch import nn, Tensor

from torchir.utils import Resampler
from torchir.transformers import Transformer


class DLIRFramework(nn.Module):
    """Deep Learning Image Registration (DLIR) Framework, a pipeline for transformers.

    The framework is an adaptation of the code presented in `De Vos 2019 <https://www.sciencedirect.com/science/article/pii/S1361841518300495>`_.

    Note: The framework differs from the original paper in the sense that DVFs are not combined by
    addition, but by composition which should result in an increase of registration precision in
    most cases `Hering et al. 2019 <https://link.springer.com/chapter/10.1007/978-3-030-32226-7_29>`_.

    """

    def __init__(self, only_last_trainable: bool = True, return_field: bool = False):
        super().__init__()
        self._resampler = Resampler(coord_dim=-1)
        self.stages = nn.ModuleList()
        self.only_last_trainable = only_last_trainable
        self.return_field = return_field

    def add_stage(self, stage: nn.Module):
        assert isinstance(stage, Transformer), "Input should be a Transformer object."
        if self.only_last_trainable and len(self.stages) > 0:
            self.stages[-1].eval()
            for param in self.stages[-1].parameters():
                param.requires_grad = False

        stage.return_field = True
        self.stages.append(stage)

    def forward(self, fixed: Tensor, moving: Tensor) -> Tensor:
        coord_field = None
        for stage in self.stages:
            if coord_field is None:
                warped = moving
            else:
                warped = self._resampler(moving, coord_field)

            coord_field = stage(fixed, warped, coord_field)

        if self.return_field:
            ret = coord_field
        else:
            ret = self._resampler(moving, coord_field)

        return ret
