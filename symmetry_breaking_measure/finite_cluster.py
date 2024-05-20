from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from diffpy.structure import Lattice, loadStructure
from pymatgen.io.cif import CifParser
from scipy.stats import norm

from symmetry_breaking_measure.base_operator import BaseOperator
from symmetry_breaking_measure.constants import ATOMIC_NUMBER


class FiniteCluster:
    """
    Define a finite cluster which contains the information of the atom sites
    (local distortions are allowed). Lattice parameters are optional. The
    users can calculate the symmetry breaking measure with respect to any local
    symmetry operation element (rotation / reflection / translation / inversion
    / screw / glide/ rotoinversion) using the information provided by a cluster
    structure.

    Properties:
    -----------
    atoms_info : np.ndarray
        atoms_info is a N by 7 array, where N is the number of atoms in the
        finite cluster.
        The first three columns are the xyz Cartesian coordinates of the atoms.
        If the structure specifies a unit cell, the atoms should include the
        periodic boundary condition.
        The 4th column is the number of electrons in this atom.
        The 5th column is the Uisoequiv of the atom. Uiso is the squared
        standard deviation of the Gaussian.
        The 6th column is the occupancy of the atom. Default to be 1.
        The 7th column is the multiplicity of the atom. Multiplicity refers to
        the number of duplicates by considering the periodic boundary
        condition. If the atom is a corner atom of a unit cell, then its
        multiplicity is 8. Default to be 1.
    lattice : Lattice
        The lattice parameters, if the finite cluster specifies a unit cell.
    space_group : int
        The space group number of the unit cell. The number should be between
        1 and 230.
    """

    def __init__(
        self,
        atoms_info: pd.DataFrame = pd.DataFrame(
            columns=["x", "y", "z", "num_electrons", "uiso", "occupancy"]
        ),
        lattice: Lattice = None,
        space_group: int = 0,
    ):
        self._atoms_info = atoms_info
        self._lattice = lattice
        self._space_group = space_group

    @property
    def atoms_info(self) -> pd.DataFrame:
        return self._atoms_info

    @property
    def lattice(self) -> Lattice:
        """
        The lattice parameters, if the finite cluster specifies a unit cell.
        """
        return self._lattice

    @property
    def space_group(self) -> int:
        """
        The space group number of the unit cell. The number should be between
        1 and 230.
        """
        return self._space_group

    @property
    def xyz(self) -> np.ndarray:
        """
        The space group number of the unit cell. The number should be between
        1 and 230.
        """
        return self._atoms_info[["x", "y", "z"]].to_numpy()

    def set_lattice(self, new_lattice: Lattice):
        """
        Set the lattice parameters for the unit cell.
        """
        self._lattice = new_lattice

    def set_space_group(self, new_space_group: int):
        """
        Set the space group number for the unit cell. Ensure the number is between 1 and 230.
        """
        if 1 <= new_space_group <= 230:
            self._space_group = new_space_group
        else:
            raise ValueError("Space group number should be between 1 and 230.")

    def set_xyz(self, new_xyz: np.ndarray):
        """
        atoms_xyz : np.ndarray
            The information of the atoms in the finite cluster. The array
            has at least three columns, which specifies the xyz Cartesian
            coordinates of the atoms.
        atoms_num_of_electron : np.ndarray
            The number of electrons of the atoms. Default to be 1.
        atoms_uiso : np.ndarray
            The number of electrons of the atoms. Default to be 1/(8pi^2).
        atoms_occupancy : np.ndarray
            The number of electrons of the atoms. Default to be 1.
        """
        # Check if the number of rows in new_xyz matches the current number of rows in self._atoms_info
        current_num_of_atoms = len(self._atoms_info.index)
        if (
            current_num_of_atoms > 0
            and new_xyz.shape[0] != current_num_of_atoms
        ):
            raise ValueError(
                f"new_xyz should have {current_num_of_atoms} rows to match the current data, but it has {new_xyz.shape[0]} rows."
            )

        self._atoms_info[["x", "y", "z"]] = new_xyz
        num_of_atoms = new_xyz.shape[0]

        if self._atoms_info["num_electrons"].isnull().all():
            self._atoms_info["num_electrons"] = np.ones((num_of_atoms, 1))
        if self._atoms_info["uiso"].isnull().all():
            self._atoms_info["uiso"] = (
                np.ones((num_of_atoms, 1)) * 1 / (8 * np.pi**2)
            )
        if self._atoms_info["occupancy"].isnull().all():
            self._atoms_info["occupancy"] = np.ones((num_of_atoms, 1))

    def set_num_electrons(
        self, new_num_electrons: Union[int, float, np.ndarray]
    ):
        """
        Set the number of electrons for atoms.
        """
        if isinstance(new_num_electrons, (float, int)):
            self._atoms_info["num_electrons"] = new_num_electrons
        else:
            if len(self._atoms_info.index) != new_num_electrons.shape[0]:
                raise ValueError(
                    "The number of rows in new_num_electrons must match the number of atoms."
                )
            self._atoms_info["num_electrons"] = new_num_electrons

    def set_uiso(self, new_uiso: Union[int, float, np.ndarray]):
        """
        Set the isotropic atomic displacement parameter for atoms.
        """
        if isinstance(new_uiso, (float, int)):
            self._atoms_info["uiso"] = new_uiso
        else:
            if len(self._atoms_info.index) != new_uiso.shape[0]:
                raise ValueError(
                    "The number of rows in new_uiso must match the number of atoms."
                )
            self._atoms_info["uiso"] = new_uiso

    def set_occupancy(self, new_occupancy: Union[int, float, np.ndarray]):
        """
        Set the occupancy for atoms.
        """
        if isinstance(new_occupancy, (float, int)):
            self._atoms_info["occupancy"] = new_occupancy
        else:
            if len(self._atoms_info.index) != new_occupancy.shape[0]:
                raise ValueError(
                    "The number of rows in new_occupancy must match the number of atoms."
                )
            self._atoms_info["occupancy"] = new_occupancy

    def import_unit_cell_from_cif(self, cif_directory: str) -> None:
        """
        Load the structure information from the cif file specified by the
        given directory.

        Parameters:
        -----------
        cif_directory : str
            The directory of the cif file.
        """

        # Parsing the cif file
        parser = CifParser(cif_directory)
        structure_df = parser.as_dict()
        structure_df = structure_df[list(structure_df.keys())[0]]
        self._space_group = structure_df["_symmetry_Int_Tables_number"]

        structure = loadStructure(cif_directory)
        self._lattice = structure.lattice
        self.set_xyz(structure.xyz_cartn)
        self.set_num_electrons(
            self.calc_num_electrons(list(structure.element))
        )
        self.set_uiso(structure.Uisoequiv)
        self.set_occupancy(structure.occupancy)

    @staticmethod
    def calc_num_electrons(symbol_list: List[str]) -> np.ndarray:
        """
        Calculate the number of electrons contained in each of the given atom
        or ion.

        Parameters:
        -----------
        symbol_list : List[str]
            A list of strings, each of which specifies the atom species and
            the number of charges in the particle if any, e.g. "Ca2+", "H",
            etc.

        Returns:
        --------
        num_of_electrons : np.ndarray
            An array of float numbers, each of which represent the number of
            electrons in the given particle.
        """
        num_electrons = np.zeros(len(symbol_list))
        for i, symbol in enumerate(symbol_list):
            if symbol[-1] not in ["+", "-"]:
                num_electrons[i] = ATOMIC_NUMBER[symbol]
            else:
                species = "".join([s for s in symbol if s.isalpha()])
                species_val = ATOMIC_NUMBER[species]
                charges = "".join([s for s in symbol if s.isdigit()])
                charges = float(charges) if charges else 1
                sign = 1 if symbol[-1] == "+" else -1
                num_electrons[i] = species_val - sign * charges
        return num_electrons

    def frac_to_cart(self, frac_xyz: np.ndarray) -> np.ndarray:
        """
        Change the atom sites from fractional coordinates to Cartesian
        coordinates.

        Parameters:
        -----------
        atoms_info: np.ndarray
            The information of the atoms in the finite cluster. The array
            has at least three columns, which specifies the xyz fractional
            coordinates of the atoms.

        Returns:
        --------
        cart_atoms_info: np.ndarray
            Same as atoms_info, except the first three columns has been
            changed from fractional coordinates to Cartesian coordinates.
        """
        cart_xyz = np.matmul(frac_xyz, self._lattice.stdbase)
        return cart_xyz

    def cart_to_frac(self, cart_xyz: np.ndarray) -> np.ndarray:
        """
        Change the atom sites from Cartesian coordinates to fractional
        coordinates.

        Parameters:
        -----------
        atoms_info: np.ndarray
            The information of the atoms in the finite cluster. The array
            has at least three columns, which specifies the xyz Cartesian
            coordinates of the atoms.

        Returns:
        --------
        frac_atoms_info: np.ndarray
            Same as atoms_info, except the first three columns has been
            changed from Cartesian coordinates to fractional coordinates.
        """
        frac_xyz = np.matmul(cart_xyz, np.linalg.inv(self._lattice.stdbase))
        return frac_xyz

    def generate_samples(
        self,
        num_samples: int,
        atoms_info: pd.DataFrame = None,
        random_seed: int = 0,
    ) -> np.ndarray:
        """
        Generate random samples according to the electron density of the
        structure specified by atoms_info. Each atom is represented by a 3d
        spherical Gaussian distribution with variance equal to the Uiso of
        this atom.

        Parameters:
        -----------
        num_of_samples: int
            The number of samples generated.
        random_seed: int
            The random seed for the numpy random number generator. If not
            specified, it defaults to 0.

        Returns:
        --------
        samples: np.ndarray
            A num_of_samples x 3 array, where each row is a random sample
            generated by the Gaussian mixture model specified by atoms_info.
        """
        np.random.seed(random_seed)
        if atoms_info is None:
            atoms_info = self._atoms_info

        phi = np.divide(
            np.multiply(atoms_info["num_electrons"], atoms_info["occupancy"]),
            atoms_info["uiso"],
        )
        phi_sum = np.cumsum(phi)
        phi_sum = phi_sum / phi_sum.iloc[-1]

        # Generate random numbers outside the loop
        u = np.random.uniform(size=num_samples)
        k_indices = np.searchsorted(phi_sum, u)

        # Batch generate samples using vectorized operations
        sigma = np.sqrt(atoms_info.loc[k_indices, "uiso"])
        x_samples = np.random.normal(atoms_info.loc[k_indices, "x"], sigma)
        y_samples = np.random.normal(atoms_info.loc[k_indices, "y"], sigma)
        z_samples = np.random.normal(atoms_info.loc[k_indices, "z"], sigma)

        # Combine samples into a single array
        samples = np.column_stack((x_samples, y_samples, z_samples))
        return samples

    def _calc_atoms_info_transformed(
        self,
        operator: BaseOperator,
        atoms_info: pd.DataFrame = None,
        **operator_kwrgs,
    ) -> pd.DataFrame:
        if atoms_info is None:
            atoms_info = self._atoms_info
        atoms_info_transformed = atoms_info.copy()
        atoms_info_transformed[["x", "y", "z"]] = operator.apply(
            atoms_xyz=atoms_info[["x", "y", "z"]].to_numpy(), **operator_kwrgs
        )
        return atoms_info_transformed

    def _calc_atoms_info_averaged(
        self,
        operator: BaseOperator,
        atoms_info: pd.DataFrame = None,
        **operator_kwrgs,
    ) -> [pd.DataFrame, pd.DataFrame]:
        if atoms_info is None:
            atoms_info = self._atoms_info
        atoms_info_transformed = self._calc_atoms_info_transformed(
            operator=operator,
            atoms_info=atoms_info,
            **operator_kwrgs,
        )
        atoms_info_averaged = atoms_info.copy()
        atoms_info_averaged = atoms_info_averaged.append(
            atoms_info_transformed, ignore_index=True
        )  # Ignoring index to avoid potential duplicate index issues
        return atoms_info_transformed, atoms_info_averaged

    def calc_symmetry_breaking_measure(
        self,
        num_samples: Union[int, list],
        operator: BaseOperator,
        method: str,
        atoms_info: pd.DataFrame = None,
        random_seed: int = 0,
        **operator_kwrgs,
    ) -> float:
        if atoms_info is None:
            atoms_info = self._atoms_info

        if method == "KL":
            atoms_info_transformed = self._calc_atoms_info_transformed(
                operator=operator,
                atoms_info=atoms_info,
                **operator_kwrgs,
            )
            D = self.calc_kl_divergence(
                num_samples=num_samples,
                atoms_info_transformed=atoms_info_transformed,
                atoms_info=atoms_info,
                random_seed=random_seed,
            )
            return D

        elif method == "JS":
            (
                atoms_info_transformed,
                atoms_info_averaged,
            ) = self._calc_atoms_info_averaged(
                operator=operator,
                atoms_info=atoms_info,
                **operator_kwrgs,
            )

            D_PM = self.calc_kl_divergence(
                num_samples=num_samples[0],
                atoms_info_transformed=atoms_info_averaged,
                atoms_info=atoms_info,
                random_seed=random_seed,
            )

            D_QM = self.calc_kl_divergence(
                num_samples=num_samples[1],
                atoms_info_transformed=atoms_info_averaged,
                atoms_info=atoms_info_transformed,
                random_seed=random_seed,
            )

            return 0.5 * D_PM + 0.5 * D_QM

    def calc_symmetry_breaking_measure_sample_size(
        self,
        num_samples: int,
        operator: BaseOperator,
        confidence_interval,
        tolerance_single_side,
        method: str,
        atoms_info: pd.DataFrame = None,
        random_seed: int = 0,
        **operator_kwrgs,
    ) -> float:
        if atoms_info is None:
            atoms_info = self._atoms_info

        if method == "KL":
            atoms_info_transformed = self._calc_atoms_info_transformed(
                operator=operator,
                atoms_info=atoms_info,
                **operator_kwrgs,
            )

            measure, sample_values = self.calc_kl_divergence(
                num_samples=num_samples,
                atoms_info_transformed=atoms_info_transformed,
                atoms_info=atoms_info,
                random_seed=random_seed,
                return_sample_values=True,
            )

            sample_size = self.calc_sample_size(
                sample_values=sample_values,
                confidence_interval=confidence_interval,
                tolerance_single_side=tolerance_single_side,
            )
            return sample_size, measure

        if method == "JS":
            (
                atoms_info_transformed,
                atoms_info_averaged,
            ) = self._calc_atoms_info_averaged(
                operator=operator,
                atoms_info=atoms_info,
                **operator_kwrgs,
            )

            measure1, sample_values1 = self.calc_kl_divergence(
                num_samples=num_samples,
                atoms_info_transformed=atoms_info_averaged,
                atoms_info=atoms_info,
                random_seed=random_seed,
                return_sample_values=True,
            )
            sample_size1 = self.calc_sample_size(
                sample_values=sample_values1,
                confidence_interval=confidence_interval,
                tolerance_single_side=tolerance_single_side,
            )

            measure2, sample_values2 = self.calc_kl_divergence(
                num_samples=num_samples,
                atoms_info_transformed=atoms_info_averaged,
                atoms_info=atoms_info_transformed,
                random_seed=random_seed,
                return_sample_values=True,
            )
            sample_size2 = self.calc_sample_size(
                sample_values=sample_values2,
                confidence_interval=confidence_interval,
                tolerance_single_side=tolerance_single_side,
            )
            sample_size_overall = self.calc_sample_size(
                sample_values=sample_values1 + sample_values2,
                confidence_interval=confidence_interval,
                tolerance_single_side=tolerance_single_side,
            )
            sample_size = [sample_size1, sample_size2, sample_size_overall]
            measure = 0.5 * measure1 + 0.5 * measure2
            return sample_size, measure

    def calc_sample_size(
        self,
        sample_values: np.ndarray,
        confidence_interval,
        tolerance_single_side,
    ):
        alpha = 1 - confidence_interval
        z_value = norm.ppf(
            1 - alpha / 2
        )  # calculates the z-score at the (1-alpha/2) percentile
        st_dev = np.sqrt(np.var(sample_values, axis=0))
        sample_size = np.ceil(
            ((st_dev * z_value) / tolerance_single_side) ** 2
        )
        return int(sample_size)

    def calc_kl_divergence(
        self,
        num_samples: int,
        atoms_info_transformed: pd.DataFrame,
        atoms_info: pd.DataFrame = None,
        random_seed: int = 0,
        return_sample_values: bool = False,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Calculate the symmetry breaking measure of the given structure, where
        the original structure to be compared with is the one specified by
        atoms_info. The method uses the Monte Carlo random sampling algorithm.

        Parameters:
        -----------
        atoms_info_transformed: np.ndarray
            The structure to calculate the symmetry breaking measure of. It
            can be the original structure with local distortions, or after
            applying certain symmetry operations.
        num_of_samples: int
            Number of samples to generate.
        return_sample_values: bool
            Whether to return sample values.

        Returns:
        --------
        symmetry_breaking_measure: float
            The symmetry breaking measure from the original structure
            specified by atoms_info to the input structure specified by
            atoms_info_transformed.
        """
        if atoms_info is None:
            atoms_info = self._atoms_info

        num_atoms, std_values, phi = self._calc_dim_std_phi(
            atoms_info=atoms_info
        )
        (
            num_atoms_transformed,
            std_values_transformed,
            phi_transformed,
        ) = self._calc_dim_std_phi(atoms_info=atoms_info_transformed)
        samples = self.generate_samples(
            num_samples=num_samples,
            atoms_info=atoms_info,
            random_seed=random_seed,
        )
        # Use comprehension loop for p and sp calculations
        p = np.array(
            [
                self._calc_gaussian_pdf_3d(
                    x=samples,
                    mean=atoms_info.iloc[i, :3].to_numpy(),
                    cov_scalar=std_values[i].reshape(1, -1, 1),
                )
                for i in range(num_atoms)
            ]
        ).T
        q = np.array(
            [
                self._calc_gaussian_pdf_3d(
                    x=samples,
                    mean=atoms_info_transformed.iloc[i, :3].to_numpy(),
                    cov_scalar=std_values_transformed[i].reshape(1, -1, 1),
                )
                for i in range(num_atoms_transformed)
            ]
        ).T

        P = np.matmul(p, phi).astype(float)
        Q = np.matmul(q, phi_transformed).astype(float)
        sample_values = np.log2(P / Q) + np.log2(
            sum(phi_transformed) / sum(phi)
        )
        kl_divergence = np.mean(sample_values)

        if return_sample_values:
            return kl_divergence, sample_values

        return kl_divergence

    def _calc_dim_std_phi(
        self,
        atoms_info: pd.DataFrame = None,
    ) -> [int, np.ndarray, np.ndarray]:
        if atoms_info is None:
            atoms_info = self._atoms_info

        num_atoms = atoms_info.shape[0]

        # Convert DataFrame columns to numpy arrays
        num_electrons = atoms_info["num_electrons"].to_numpy()
        uiso = atoms_info["uiso"].to_numpy()
        occupancy = atoms_info["occupancy"].to_numpy()

        phi = np.divide(num_electrons * occupancy, uiso)
        std_values = np.sqrt(uiso)
        return num_atoms, std_values, phi

    @staticmethod
    def _calc_gaussian_pdf_3d(
        x: np.ndarray, mean: np.ndarray, cov_scalar: np.ndarray
    ) -> float:
        """
        Calculate the probability density of 3d Gaussian distributions at
        given point x and the mean, and covariance matrix of the distribution.

        Parameters:
        -----------
        x: np.ndarray
            The point we calculate the probability density at.
        mean: np.ndarray
            The mean of the spherical Gaussian distribution.
        cov_scalar: np.ndarray
            The scalar value by which the identity matrix is multiplied to obtain
            the covariance matrix.

        Returns:
        --------
        gaussian_pdf_3d: float
            The probability density of 3d Gaussian distributions at given
            point x and the mean, and covariance matrix of the distribution.
        """
        disp = x - mean
        power = -0.5 * np.sum(disp**2, axis=-1) / cov_scalar.squeeze(-1)
        denominator = np.sqrt((2 * np.pi * cov_scalar.squeeze()) ** 3)
        gaussian_pdf_3d = 1 / denominator * np.exp(power)

        return gaussian_pdf_3d
