import matplotlib.pyplot as plt
import numpy as np
from bg_mpl_stylesheet.bg_mpl_stylesheet import bg_mpl_style
from diffpy.structure import Lattice
from mpl_toolkits.mplot3d import art3d


class PlotHelper:
    @staticmethod
    def _get_lattice_component(lattice: Lattice):
        stdbase = lattice.stdbase
        a, b, c = stdbase[:, 0], stdbase[:, 1], stdbase[:, 2]
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
            v[f], facecolors="g", edgecolor="none", alpha=0.05
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
        lc = art3d.Poly3DCollection(v[e], edgecolor="g", alpha=0.1)
        return pc, lc

    @staticmethod
    def plot_samples(samples, lattice=None, save_path=None):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plt.style.use(bg_mpl_style)

        if lattice:
            pc, lc = PlotHelper._get_lattice_component(lattice)
            ax.add_collection(pc)
            ax.add_collection(lc)

        ax.scatter(
            samples[:, 0], samples[:, 1], samples[:, 2], marker=".", s=0.5
        )
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")

        plt.show()
        if save_path != None:
            fig.savefig(save_path)
        return ax

    @staticmethod
    def plot_symmetry_breaking_measure(
        x, y, xlabel, ylabel, xlim, ylim, figsize, save_path=None
    ):
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.style.use(bg_mpl_style)
        fig.set_size_inches(figsize[0], figsize[1])
        y_round = np.round(y, 5)
        idx = np.where(y_round == np.amin(y_round))

        ax.plot(x, y)
        for i in idx[0]:
            ax.axvline(x=x[i], color="g", linestyle="--")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.show()
        if save_path != None:
            fig.savefig(save_path)
        return ax

    @staticmethod
    def plot_symmetry_breaking_measure_error_bar(
        x, y, xlabel, ylabel, xlim, ylim, figsize, save_path=None
    ):
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.style.use(bg_mpl_style)
        fig.set_size_inches(figsize[0], figsize[1])
        y = np.array(y)
        plt.errorbar(x, y[:, 1], yerr=y[:, [0, 2]].T)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.show()
        if save_path != None:
            fig.savefig(save_path)
        return ax
