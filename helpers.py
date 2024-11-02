""" helper functions for the Monte Carlo simulation dataset """
import pathlib as pl
from functools import cached_property
from torch.utils.data import Dataset, Subset
import numpy as np
import pandas as pd
import pyvista as pv
import discretisedfield as df
import torch
from sklearn.model_selection import train_test_split


class MCSims(Dataset):
    """
        This is a ``torch`` DataSet class to read Sam's Monte Carlo simulation files and
        lazy load them as torch tensors for the ML task. The main methods that need to be
        implemented are the ``__getitem__`` and ``__len__`` for the dataset class to work
        with the ``DataLoader`` class.

        Example
        -------
        >>> dataset = MCSims()
        """

    def __init__(self):
        self.base_path = pl.Path(
            "/scratch/holtsamu/fege_phase_diagram/temperature_field_diagram/data/"
        )
        self.x_dim = 97
        self.y_dim = 97
        self.z_dim = 97
        self.v_dim = 3

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
        DF = pd.DataFrame(columns=["H (A/m)", "T (K)", "File"])
        for path in self.base_path.glob("data_*"):
            dat_arr = np.loadtxt(path / "log.dat")
            vti_files = list(path.glob("*.vti"))
            vti_files.sort()
            for i, file in enumerate(vti_files):
                DF.loc[len(DF)] = [dat_arr[i, 0], dat_arr[0, 1], file]

        return DF

    def __getitem__(self, index: [int, slice]) -> torch.Tensor:
        """
        Required by the ``DataLoader`` class to get the tensors corresponding to the
        simulation files. It returns either a 4 dimensional tensor or a 5 dimensional
        tensor depending on the value of index being integer or slice respectively. The
        method does this in a lazy way so that the whole data is not loaded in the memory
        at the same time.

        Example
        -------
        >>> dataset = MCSims()
        >>> dataset[10:20]
        tensor([[[[[-1.6286e-01, -7.3638e-01,  8.2156e-01,  ..., -5.6897e-01,
             6.1511e-01, -2.2316e-01],
        ...
        >>> dataset[10:20].shape
        torch.Size([10, 3, 97, 97, 97])
        """
        get_val = self.data_frame["File"][index]
        if isinstance(index, slice):
            values = np.array(
                [
                    pv.read(file)
                    .point_data["m"]
                    .reshape(self.x_dim, self.y_dim, self.z_dim, self.v_dim)
                    .transpose((3, 2, 1, 0))
                    for file in get_val
                ]
            )
            return torch.tensor(values).float()
        else:
            return torch.tensor(
                pv.read(get_val)
                .point_data["m"]
                .reshape(self.x_dim, self.y_dim, self.z_dim, self.v_dim)
                .transpose((3, 2, 1, 0))
            ).float()

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Example
        -------
        >>> dataset = MCSims()
        >>> len(dataset)
        2601
        """
        return len(self.data_frame)

    def field_from_index(self, index: int) -> df.Field:
        """
        Helper function to return the simulation results (vti files) as a
        ``discretisedfield.Field`` object. It is much easier to handle and plot Fields
        compared to the pyvista mesh objects.

        Example
        -------
        >>> dataset = MCSims()
        >>> dataset.field_from_index(77)
        Field(...)
        """
        file = self.data_frame["File"][index]
        field_pv = pv.read(file)
        value = (
            field_pv.point_data["m"]
            .reshape(self.x_dim, self.y_dim, self.z_dim, self.v_dim)
            .transpose((2, 1, 0, 3))
        )
        p1 = field_pv.points[0]
        p2 = field_pv.points[-1]
        n = (self.x_dim, self.y_dim, self.z_dim)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        return df.Field(mesh=mesh, value=value, nvdim=self.v_dim)

    def train_test_split(self, test_size=0.2, random_state=42):
        """
        Split the dataset into training and test sets.

        Parameters
        ----------
        test_size : float
            Proportion of the dataset to include in the test split (e.g., 0.2 for 20%).
        random_state : int, optional
            Controls the shuffling applied to the data before splitting.

        Returns
        -------
        train_dataset : Subset
            Training subset of the MCSims dataset.
        test_dataset : Subset
            Test subset of the MCSims dataset.
        """
        # Generate indices for the full dataset
        indices = list(range(len(self)))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )

        # Create subset datasets
        train_dataset = Subset(self, train_indices)
        test_dataset = Subset(self, test_indices)

        return train_dataset, test_dataset

    def plot_field_xy(self, index):
        """
        Helper function to plot the 3D magnetisation configuration as a holoviews plot.
        The plot shows a xy-plane cut of the 3D vector field with a slider to move along
        z-direction.

        Example
        -------
        >>> dataset = MCSims()
        >>> dataset.plot_field_xy(77)
        :DynamicMap   [z]
        """
        return self.field_from_index(index=index).hv(
            kdims=["x", "y"], scalar_kw={"clim": (-1, 1)}
        )


def plot_field_xy_from_tensor(tensor):
    """
    Helper function to convert a ``torch.Tensor`` to ``discretisedfield.Field`` and
    subsequently plot the xy-plane cut of the 3D vector field using holoviews plot with
    a slider to move along z-direction. Note: the value of mesh points are not realistic
    in this case

    Example
    -------

    >>> plot_field_xy_from_tensor(tensor)
    :DynamicMap   [z]
    """
    x_dim = 97
    y_dim = 97
    z_dim = 97
    v_dim = 3
    value = tensor.to("cpu").numpy().transpose((1, 2, 3, 0))
    p1 = (0, 0, 0)
    p2 = (x_dim - 1, y_dim - 1, z_dim - 1)
    mesh = df.Mesh(p1=p1, p2=p2, n=(x_dim, y_dim, z_dim))
    return df.Field(mesh=mesh, value=value, nvdim=v_dim).hv(
        kdims=["x", "y"], scalar_kw={"clim": (-1, 1)}
    )
