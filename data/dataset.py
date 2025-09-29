"""helper functions for the Monte Carlo simulation dataset"""

import pathlib as pl
import random
from functools import cached_property

import discretisedfield as df
import numpy as np
import pandas as pd
import pyvista as pv
import torch
from torch.utils.data import Dataset


class RotateTransform:
    """randomly rotate the tensor by 90, 180, or 270 degrees along the z-axis"""

    def __call__(self, x):
        k = random.randint(0, 3)
        return torch.rot90(x, k, dims=(1, 2))


class MCSims(Dataset):
    """
    Torch DataSet class to read Sam's Monte Carlo simulation files with early loading.
    The dataset is saved to disk to speed up repeated training runs.

    Parameters
    ----------
    preload : bool, optional
        If True, load the dataset into memory during initialization.
    preprocess : bool, optional
        If True, apply preprocessing to tensors before caching.
    transform : bool, optional
        If True, apply a random rotation to the tensors along the z-axis.

    Example
    -------
    >>> dataset = MCSims()
    """

    def __init__(self, preload=True, preprocess=True, augment=True):
        self.base_path = pl.Path(
            "/scratch/holtsamu/fege_phase_diagram/temperature_field_diagram/data/"
        )
        self.cache_file = pl.Path("data/cache.pt")
        self.tensor_cache = pl.Path("data/tensor_cache.pt")
        self.preload = preload
        self.preprocess = preprocess
        self.augment = augment

        self.x_dim = 97
        self.y_dim = 97
        self.z_dim = 97
        self.v_dim = 3

        if preload:
            self.preloaded_data = self._load_tensor_cache()
        else:
            self.preloaded_data = None

    def _load_tensor_cache(self):
        """Load the entire dataset into memory if a cached tensor file exists."""
        if self.tensor_cache.exists():
            return torch.load(self.tensor_cache)
        return None

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        """
        The ``data_frame`` property relates external field (H) and temperature (T) values
        with the corresponding simulation files using ``pandas.DataFrame``. This will be
        useful to plot the phase diagram in the end where one can simply add a "Class"
        column to the DataFrame post clustering.

        Example
        -------
        >>> dataset = MCSims()
        >>> dataset.data_frame
                    H (A/m)   T (K)                                               File
        0          0.000000  2280.0  /scratch/holtsamu/fege_phase_diagram/temperatu...
        1       7957.747150  2280.0  /scratch/holtsamu/fege_phase_diagram/temperatu...
        2      15915.494301  2280.0  /scratch/holtsamu/fege_phase_diagram/temperatu...
        """
        if self.cache_file.exists():
            return pd.read_csv(self.cache_file)
        else:
            DF = pd.DataFrame(columns=["H (A/m)", "T (K)", "File"])
            for path in self.base_path.glob("data_*"):
                dat_arr = np.loadtxt(path / "log.dat")
                vti_files = list(path.glob("*.vti"))
                vti_files.sort()
                for i, file in enumerate(vti_files):
                    DF.loc[len(DF)] = [dat_arr[i, 0], dat_arr[0, 1], file]

            DF.to_csv(self.cache_file, index=False)
            return DF

    def _load_tensor(self, file_path: str) -> torch.Tensor:
        """Helper function to load a single tensor from a .vti file."""
        tensor = torch.tensor(
            pv.read(file_path)
            .point_data["m"]
            .reshape(self.x_dim, self.y_dim, self.z_dim, self.v_dim)
            .transpose((3, 2, 1, 0))
        )

        if self.preprocess:
            tensor = self._preprocess_tensor(tensor)

        return tensor

    def _preprocess_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess the tensor data.
        apply adaptive average pooling to downsample the tensor and reduce noise.
        Input tensor shape: (3, 97, 97, 97)
        Output tensor shape: (3, 48, 48, 48)
        """
        return torch.nn.AvgPool3d(kernel_size=7, stride=2, padding=2)(tensor)

    def __getitem__(self, index: int | slice) -> torch.Tensor:
        """
        Fetch simulation data as tensors.
        If preload=True, load from memory instead of reading files.
        """
        k = index % 4  # which rotation to apply
        index = (
            index // 4 if self.augment else index
        )  # if augment, get the original sample index

        if self.preload and self.preloaded_data is not None:
            tensor = self.preloaded_data[index]
        else:
            get_val = self.data_frame["File"][index]

            if isinstance(index, slice):
                values = np.array([self._load_tensor(file) for file in get_val])
                tensor = torch.tensor(values)
            else:
                tensor = self._load_tensor(get_val)

        return torch.rot90(tensor, k, dims=(1, 2)) if self.augment else tensor

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Example
        -------
        >>> dataset = MCSims()
        >>> len(dataset)
        2601
        """
        if self.augment:
            return len(self.data_frame) * 4
        else:
            return len(self.data_frame)

    def save_tensor_cache(self):
        """Preload all tensors and save them to disk for faster access."""
        all_tensors = torch.stack(
            [self._load_tensor(file) for file in self.data_frame["File"]]
        )
        torch.save(all_tensors, self.tensor_cache)

    def field_from_index(self, index: int) -> df.Field:
        """
        Helper function to return the simulation results as a
        ``discretisedfield.Field`` object.
        """
        value = self.preloaded_data[index].numpy().transpose((1, 2, 3, 0))
        p1 = (0, 0, 0)
        p2 = (self.x_dim - 1, self.y_dim - 1, self.z_dim - 1)
        mesh = df.Mesh(p1=p1, p2=p2, n=(self.x_dim, self.y_dim, self.z_dim))
        return df.Field(mesh=mesh, value=value, nvdim=self.v_dim)

    def plot_field_xy(self, index):
        """
        Helper function to plot the 3D magnetisation configuration.
        """
        return self.field_from_index(index=index).hv(
            kdims=["x", "y"], scalar_kw={"clim": (-1, 1)}
        )


if __name__ == "__main__":
    # dataset = MCSims(preload=False)
    # dataset.save_tensor_cache()
    # print("Tensor cache saved")
    pass
