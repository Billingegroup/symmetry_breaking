from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from bg_mpl_stylesheet.bg_mpl_stylesheet import bg_mpl_style
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imread
from mpl_toolkits.mplot3d import art3d

from symmetry_breaking_measure.finite_cluster import FiniteCluster
from symmetry_breaking_measure.constants import (
    CA_NUM_ELECTRON,
    CA_UISO,
    O_NUM_ELECTRON,
    O_UISO,
    PEROVSKITE_EXTRA_BOUND_FRAC,
    PEROVSKITE_GLAZER_SYSTEM,
    PEROVSKITE_LATTICE_A,
    PEROVSKITE_PRECISION,
    TI_NUM_ELECTRON,
    TI_UISO,
)

BLUE, RED, YELLOW, MIDDLE = "#0B3C5D", "#B82601", "#D9B310", "#a8b6c1"


class Perovskite(
    FiniteCluster
):  # pylint: disable=too-many-instance-attributes
    """
    Represents a cluster of nickel atoms arranged in different geometric forms.

    This class provides functionality to create a finite cluster of nickel atoms
    in either an ellipsoidal or rectangular solid shape. It also allows for plotting
    the resulting cluster using various visualization methods.
    """

    def __init__(self, amplitude: float, cif_directory: str) -> None:
        super().__init__()
        self.import_unit_cell_from_cif(cif_directory)
        self.amplitude = amplitude

    def plot_perovskite(  # pylint:disable=26/15) (too-many-locals)
        self,
        projection_plot=True,
        plot_3d=False,
        plot_3d_import=None,
        filename=None,
        style=None,
    ):
        # Define the centers and radii of the spheres
        centers = self.xyz
        radii = np.sqrt(self.atoms_info["uiso"])
        num_electrons = self._atoms_info["num_electrons"]
        electron_to_color = {
            CA_NUM_ELECTRON: YELLOW,
            TI_NUM_ELECTRON: BLUE,
            O_NUM_ELECTRON: RED,
        }
        # Using a list comprehension to generate the colors
        colors = [
            electron_to_color.get(num, "default_color")
            for num in num_electrons
        ]

        # Adjust the figure size based on the plots being displayed
        if (plot_3d or plot_3d_import) and projection_plot:
            fig_size = (12, 3)
        elif plot_3d:
            fig_size = (5, 5)
        elif projection_plot:
            fig_size = (9, 3)
        else:
            fig_size = (5, 5)
        # Default to 3D plot size if no arguments are True
        # Create a 3D figure with the adjusted size
        fig = plt.figure(figsize=fig_size)
        if style:
            plt.style.use(style)

        if plot_3d:
            if projection_plot:
                ax = fig.add_subplot(141, projection="3d")
            else:
                ax = fig.add_subplot(111, projection="3d")

            # Plot each sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            for center, radius, color in zip(centers, radii, colors):
                x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z = center[2] + radius * np.outer(
                    np.ones(np.size(u)), np.cos(v)
                )
                ax.plot_surface(x, y, z, color=color, alpha=0.6)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            ax.set_title("3D Pervoskite Visualization")

        if plot_3d_import:
            if projection_plot:
                ax = fig.add_subplot(141)
            else:
                ax = fig.add_subplot(111)
            img = imread(plot_3d_import)
            ax.imshow(img)
            ax.set_axis_off()

            ax.set_title("3D Pervoskite Visualization")

            # If projection_plot is True, create the projection plots
        if projection_plot:
            # Projection on the XY plane
            ax_xy = fig.add_subplot(
                142 if (plot_3d or plot_3d_import) else 131
            )
            ax_xy.scatter(
                [c[0] for c in centers],
                [c[1] for c in centers],
                c=colors,
                alpha=0.75,
            )
            ax_xy.set_aspect("equal", adjustable="box")
            ax_xy.set_title("Projection on ab plane")
            ax_xy.set_xlabel("a-axis")
            ax_xy.set_ylabel("b-axis")

            # Projection on the XZ plane
            ax_xz = fig.add_subplot(
                143 if (plot_3d or plot_3d_import) else 132
            )
            ax_xz.scatter(
                [c[0] for c in centers],
                [c[2] for c in centers],
                c=colors,
                alpha=0.75,
            )
            ax_xz.set_aspect("equal", adjustable="box")
            ax_xz.set_title("Projection on ac plane")
            ax_xz.set_xlabel("a-axis")
            ax_xz.set_ylabel("c-axis")

            # Projection on the YZ plane
            ax_yz = fig.add_subplot(
                144 if (plot_3d or plot_3d_import) else 133
            )
            ax_yz.scatter(
                [c[1] for c in centers],
                [c[2] for c in centers],
                c=colors,
                alpha=0.75,
            )
            ax_yz.set_aspect("equal", adjustable="box")
            ax_yz.set_title("Projection on bc plane")
            ax_yz.set_xlabel("b-axis")
            ax_yz.set_ylabel("c-axis")

            plt.tight_layout()

        # Display the plot
        plt.show()
        if filename:
            fig.savefig(filename)

    def expand_by_supercell(self, num_supercells: int):
        x_range, y_range, z_range = self.calc_range(num_supercells)
        self.expand_by_range(x_range, y_range, z_range)

    def calc_num_each_species(self, actual=True, ratio=False):
        # Count the number of rows for each species
        ca_count = len(
            self._atoms_info[
                self._atoms_info["num_electrons"] == CA_NUM_ELECTRON
            ]
        )
        ti_count = len(
            self._atoms_info[
                self._atoms_info["num_electrons"] == TI_NUM_ELECTRON
            ]
        )
        o_count = len(
            self._atoms_info[
                self._atoms_info["num_electrons"] == O_NUM_ELECTRON
            ]
        )

        # Return the counts as ratios
        total_count = ca_count + ti_count + o_count
        if ratio:
            return np.array([ca_count, ti_count, o_count]) / total_count
        if actual:
            return np.array([ca_count, ti_count, o_count])

    def calc_range(self, num_supercells: int):
        lattice_params = [self._lattice.a, self._lattice.b, self._lattice.c]
        return [
            [
                -PEROVSKITE_EXTRA_BOUND_FRAC * param,
                num_supercells * param + PEROVSKITE_EXTRA_BOUND_FRAC * param,
            ]
            for param in lattice_params
        ]

    def _remove_duplicates_with_rounding(
        self, df, precision=PEROVSKITE_PRECISION
    ):
        # Store the original values
        df_rounded = df.copy()
        df_rounded["x"] = df_rounded["x"].round(precision)
        df_rounded["y"] = df_rounded["y"].round(precision)
        df_rounded["z"] = df_rounded["z"].round(precision)

        # Drop duplicates based on the rounded values
        df_rounded = df_rounded.drop_duplicates(subset=["x", "y", "z"])

        # Use the indices of the rows kept in the temporary DataFrame
        # to select the corresponding rows from the original DataFrame
        df = df.loc[df_rounded.index]

        return df

    def expand_by_range(self, x_range: list, y_range: list, z_range: list):
        # Helper function to expand atoms in a given direction
        def expand_direction(atoms, direction, direction_range, lattice_param):
            new_atoms = atoms.copy(deep=True)
            # The maximum number of iterations is the range divided by lattice parameter
            max_iterations = int(
                (direction_range[1] - direction_range[0]) / lattice_param
            )
            for _ in range(max_iterations):
                # Add atoms by adding lattice parameter
                atoms_plus = atoms.copy(deep=True)
                atoms_plus[direction] += lattice_param
                new_atoms = new_atoms.append(atoms_plus, ignore_index=True)

                # Subtract atoms by subtracting lattice parameter
                atoms_minus = atoms.copy(deep=True)
                atoms_minus[direction] -= lattice_param
                new_atoms = new_atoms.append(atoms_minus, ignore_index=True)

                # Update atoms for next iteration
                atoms = new_atoms.copy(deep=True)

            # Filter atoms to only those within the given range
            new_atoms = new_atoms[
                new_atoms[direction].between(
                    direction_range[0], direction_range[1]
                )
            ]
            # Drop duplicates based on x, y, z coordinates
            new_atoms = self._remove_duplicates_with_rounding(new_atoms)
            return new_atoms

        # Expand in x direction
        expanded_atoms = expand_direction(
            self._atoms_info, "x", x_range, self._lattice.a
        )
        # Expand in y direction
        expanded_atoms = expand_direction(
            expanded_atoms, "y", y_range, self._lattice.b
        )
        # Expand in z direction
        expanded_atoms = expand_direction(
            expanded_atoms, "z", z_range, self._lattice.c
        )

        # Update the atoms info to the expanded atoms
        self._atoms_info = expanded_atoms.reset_index(drop=True)
