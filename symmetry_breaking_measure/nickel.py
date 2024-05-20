from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from bg_mpl_stylesheet.bg_mpl_stylesheet import bg_mpl_style
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import art3d

from symmetry_breaking_measure import FiniteCluster
from symmetry_breaking_measure.constants import (
    NICKEL_LATTICE,
    NICKEL_NUM_ELECTRON,
    NICKEL_UNIT_CELL_ATOMS,
)


class Nickel(FiniteCluster):  # pylint: disable=too-many-instance-attributes
    """
    Represents a cluster of nickel atoms arranged in different geometric forms.

    This class provides functionality to create a finite cluster of nickel atoms
    in either an ellipsoidal or rectangular solid shape. It also allows for plotting
    the resulting cluster using various visualization methods.
    """

    def __init__(
        self, cutoff: str, cutout_x: float, cutout_y: float, cutout_z: float
    ) -> None:
        super().__init__()
        self._cutoff = cutoff
        self._cutout_x = cutout_x
        self._cutout_y = cutout_y
        self._cutout_z = cutout_z
        self._lattice = NICKEL_LATTICE
        self._unitcell_atoms = NICKEL_UNIT_CELL_ATOMS
        # Create a meshgrid for the supercell
        x_range = np.arange(np.floor(-2 * cutout_x), np.ceil(2 * cutout_x) + 1)
        y_range = np.arange(np.floor(-2 * cutout_y), np.ceil(2 * cutout_y) + 1)
        z_range = np.arange(np.floor(-2 * cutout_z), np.ceil(2 * cutout_z) + 1)
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        # Reshape and stack the meshgrid to generate supercell_round
        supercell_round = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
        # Combine unitcell_atoms with supercell_round using broadcasting
        expanded_unitcell = self._unitcell_atoms[:, np.newaxis, :]
        supercell_atoms = (expanded_unitcell + supercell_round).reshape(-1, 3)
        self._supercell_atoms = np.unique(supercell_atoms, axis=0)

        if self._cutoff == "ellipsoid":
            # Ellipsoid cut off from Nickel
            self.get_ellipsoid()
        elif self._cutoff == "rectangular_solid":
            # Rectangular cut off from Nickel
            self.get_rectangular_solid()

    def get_ellipsoid(self):
        """
        Create an ellipsoid cluster of nickel atoms.

        Returns:
        --------
        ellipsoid_cluster : FiniteCluster
            Ellipsoidal cluster of nickel atoms.
        """
        squared_coord = np.square(self._supercell_atoms)
        mask = (
            squared_coord[:, 0] / self._cutout_x**2
            + squared_coord[:, 1] / self._cutout_y**2
            + squared_coord[:, 2] / self._cutout_z**2
            <= 1
        )
        ellipsoid_atoms_frac = self._supercell_atoms[mask]
        ellipsoid_atoms_cart = self.frac_to_cart(ellipsoid_atoms_frac)
        self.set_xyz(ellipsoid_atoms_cart)
        self.set_num_electrons(NICKEL_NUM_ELECTRON)

    def get_rectangular_solid(self):
        """
        Create a rectangular solid cluster of nickel atoms.

        Returns:
        --------
        rectangular_cluster : FiniteCluster
            Rectangular cluster of nickel atoms.
        """
        mask = (
            (
                (self._supercell_atoms[:, 0] >= 0)
                & (self._supercell_atoms[:, 0] <= self._cutout_x)
            )
            & (
                (self._supercell_atoms[:, 1] >= 0)
                & (self._supercell_atoms[:, 1] <= self._cutout_y)
            )
            & (
                (self._supercell_atoms[:, 2] >= 0)
                & (self._supercell_atoms[:, 2] <= self._cutout_z)
            )
        )
        rectangular_solid_atoms_frac = self._supercell_atoms[mask]
        rectangular_solid_atoms_cart = self.frac_to_cart(
            rectangular_solid_atoms_frac
        )
        self.set_xyz(rectangular_solid_atoms_cart)
        self.set_num_electrons(NICKEL_NUM_ELECTRON)

    def plot_ellipsoid(
        self, ax: plt.Axes, num_samples: int, cmap: LinearSegmentedColormap
    ) -> plt.Axes:
        """
        Plot the ellipsoid cluster of nickel atoms.

        Returns:
        --------
        ax : plt.Axes
            Axes.
        """
        samples = self.generate_samples(num_samples)
        plt.style.use(bg_mpl_style)

        def _plot_ellipsoid_surface(
            ax: plt.Axes, cmap: LinearSegmentedColormap
        ) -> None:
            """Plot the ellipsoid surface."""
            r = 1 * self._lattice.a
            u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
            x = self._cutout_x * r * np.cos(u) * np.sin(v)
            y = self._cutout_y * r * np.sin(u) * np.sin(v)
            z = self._cutout_z * r * np.cos(v)
            ax.plot_surface(x, y, z, cmap=cmap, alpha=0.2)

        def _plot_ellipsoid_vertices(
            ax: plt.Axes, cmap: LinearSegmentedColormap
        ) -> None:
            """Plot the vertices for the ellipsoid."""
            vertices = np.array(
                [
                    [0, 0, 0],
                    [0, self._cutout_y, 0],
                    [0, 0, self._cutout_z],
                    [0, -self._cutout_y, 0],
                    [0, 0, -self._cutout_z],
                    [self._cutout_x, 0, 0],
                    [-self._cutout_x, 0, 0],
                ]
            )
            vertices = self.frac_to_cart(vertices)
            c = cmap(0)
            for i in range(1, len(vertices)):
                ax.plot(
                    vertices[[0, i], 0],
                    vertices[[0, i], 1],
                    vertices[[0, i], 2],
                    color=c,
                    linestyle="--",
                    alpha=0.5,
                )

        ax.scatter(
            samples[:, 0],
            samples[:, 1],
            samples[:, 2],
            c=np.linalg.norm(samples, axis=1),
            marker=".",
            s=0.5,
            cmap=cmap,
        )
        _plot_ellipsoid_surface(ax, cmap)
        _plot_ellipsoid_vertices(ax, cmap)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        return ax

    def plot_rectangular_solid(  # pylint:disable=too-many-arguments
        self,
        ax: plt.Axes,
        num_samples: int,
        cmap: LinearSegmentedColormap,
        facecolor=Any,
        in_plane: bool = None,
        center: bool = None,
    ):
        """
        Plot the rectangular solid cluster of nickel atoms.

        Returns:
        --------
        ax : plt.Axes
            Axes.
        """
        samples = self.generate_samples(num_samples)
        plt.style.use(bg_mpl_style)

        def _plot_rectangular_solid_surface(
            ax: plt.Axes,
            samples: np.ndarray,
            cmap: LinearSegmentedColormap,
            in_plane=None,
            center=None,
        ) -> None:
            """Plot the surface of the rectangular solid."""
            a_center, b_center, c_center = (
                0.5 * self.lattice.a,
                0.5 * self.lattice.b,
                0.5 * self.lattice.c,
            )
            if in_plane is True:
                c = np.where(np.abs(samples[:, 2] - b_center) < 1, 1, 0)
            elif in_plane is False:
                c = np.where(np.abs(samples[:, 2] - b_center) < 1, 0, 1)
            elif center is True:
                c = np.where(
                    np.any(np.abs(samples - 1.76) < 1, axis=1), 1, 0.2
                )
            elif center is False:
                c = np.where(
                    np.any(np.abs(samples - 1.76) < 1, axis=1), 0.2, 1
                )
            else:
                c = np.linalg.norm(
                    samples - [a_center, b_center, c_center], axis=1
                )
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                samples[:, 2],
                c=c,
                marker=".",
                s=0.5,
                cmap=cmap,
            )

        def _plot_rectangular_solid_edges(ax, facecolor):
            """Plot the edges of the rectangular solid."""
            edges = np.array(
                [
                    [self._cutout_x, 0, 0],
                    [0, self._cutout_y, 0],
                    [0, 0, self._cutout_z],
                ]
            )
            edges = self.frac_to_cart(edges)
            a, b, c = edges[:, 0], edges[:, 1], edges[:, 2]
            v = np.array([[0, 0, 0], a, a + b, b, c, a + c, a + b + c, b + c])
            f = np.array(
                [
                    [0, 2, 1],
                    [0, 3, 2],
                    [1, 2, 6],
                    [1, 6, 5],
                    [0, 5, 4],
                    [0, 1, 5],
                    [4, 5, 6],
                    [6, 7, 4],
                    [3, 7, 6],
                    [6, 2, 3],
                    [0, 4, 7],
                    [7, 3, 0],
                ]
            )
            pc = art3d.Poly3DCollection(
                v[f], facecolors=facecolor, edgecolor="none", alpha=0.1
            )
            e = np.array(
                [
                    [0, 1],
                    [0, 3],
                    [0, 4],
                    [1, 2],
                    [1, 5],
                    [4, 5],
                    [4, 7],
                    [3, 2],
                    [3, 7],
                    [2, 6],
                    [5, 6],
                    [7, 6],
                ]
            )
            lc = art3d.Poly3DCollection(v[e], edgecolor=facecolor, alpha=0.1)
            ax.add_collection(pc)
            ax.add_collection(lc)

        _plot_rectangular_solid_surface(
            ax, samples, cmap, in_plane=in_plane, center=center
        )
        _plot_rectangular_solid_edges(ax, facecolor)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        return ax
