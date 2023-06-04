"""Plotting Module."""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from collections import namedtuple
from itertools import combinations
from scipy.stats import gaussian_kde
from scipy.spatial import distance_matrix

from combust.utils.calculator import coord_num
from combust.utils.rxn_data import rxn_dict


def get_dataframe_columns_labels(nums):
    """Get column labels of data-frame.

    Parameters
    ----------
    nums : ndarray, shape=(N)
        The 1D array of atomic numbers for N atoms.

    """
    if nums.ndim != 1:
        raise ValueError(f"Argument num with shape={nums.shape} is not a 1D array!")
    num2sym = {1: 'H', 8: 'O'}
    symbols = []
    count_h, count_o = 1, 1
    for num in nums:
        if num not in num2sym.keys():
            raise ValueError(f"Atomic number {num} not among supported {num2sym.keys()} numbers!")
        element = num2sym[num]
        if element == "H":
            symbols.append(f"{element}{count_h}")
            count_h += 1
        elif element == "O":
            symbols.append(f"{element}{count_o}")
            count_o += 1
        else:
            raise ValueError(f"Element {element} not covered!")
    columns = [f"{e1}_{e2}" for e1, e2 in combinations(symbols, 2)]
    return columns


def get_pairwise_distance_triu(coords):
    """Return upper triangular Minkowski p-norm distance matrix.

    Parameters
    ----------
    coords: ndarray, shape=(N * 3)
        The 1D array of flattened atomic coordinates for N atoms.

    """
    if coords.ndim != 1:
        raise ValueError(f"Argument coords with shape={coords.shape} is not a 1D array!")
    # reshape coordinates to have 3 columns (then number of rows would be number of atoms)
    coords = coords.reshape(-1, 3)
    dist = distance_matrix(coords, coords, p=2)
    return dist[np.triu_indices(len(coords), k=1)]


def get_pairwise_distance_dataframe(nums, coords):
    """Return pair-wise distance data frame.

    Parameters
    ----------
    nums : ndarray, shape=(N)
        The 1D array of atomic numbers for N atoms.
    coords: ndarray, shape=(M, N * 3)
        The 2D array of flattened atomic coordinates for M data points and N atoms.

    """
    columns = get_dataframe_columns_labels(nums)
    dist = np.apply_along_axis(get_pairwise_distance_triu, 1, coords)
    dist = pd.DataFrame(dist, columns=columns)
    assert len(coords) == len(dist)
    return dist


def get_cn_arrays(nums, coords, cn_labels, mu, sigma):
    """Return coordination number array for M data points.

    Parameters
    ----------
    nums : ndarray, shape=(M, N)
        The 2D array of atomic numbers for M data points and N atoms
    coords: ndarray, shape=(M, N, 3)
        The 3D array of atomic coordinates for M data points and N atoms.
    cn_labels: sequence of str
        List of desired coordination number labels.
    mu : float
        The mu value used for computing coordination number.
    sigma : float
        The sigma value used for computing coordination number.

    """
    # check nums & coords shape
    if nums.shape[0] != coords.shape[0]:
        raise ValueError("Argument nums & coords do not represent the same number of data points!")
    if nums.shape[1] != coords.shape[1]:
        raise ValueError("Argument nums & coords do not represent the same number of atoms!")
    # get unique atomic numbers
    nums = np.unique(nums, axis=0)
    if len(nums) != 1:
        raise ValueError(f"Cannot compute coordination number b/c more than one rxn found: {nums}")
    nums = nums.flatten()

    # change coords to 2D array with shape=(M, N * 3)
    coords = coords.reshape(coords.shape[0], coords.shape[1] * coords.shape[2])

    # compute data frame & coordination number array
    dist = get_pairwise_distance_dataframe(nums, coords)
    cn = coord_num([getattr(dist, item).values for item in cn_labels], mu=mu, sigma=sigma)
    assert len(cn) == len(coords)
    return cn


def get_rxn_namedtuple(cn1_array, cn2_array, energy):
    """Make NamedTuple for plotting.

    Parameters
    ----------
    cn1_array : ndarray, shape=(M,)
        The 1D array of 1st coordination number for M data points.
    cn2_array : ndarray, shape=(M,)
        The 1D array of 2nd coordination number for M data points.
    energy : np.ndarray, shape=(M,)
        The 1D array for energy values for M data points.

    """
    # check argument shapes
    if cn1_array.ndim != 1 or cn2_array.ndim != 1 or energy.ndim != 1:
        raise ValueError("Arguments cn1_array, cn2_array, & energy should all be 1D arrays.")
    if cn1_array.shape != cn2_array.shape != energy.shape:
        raise ValueError("Arguments cn1_array, cn2_array, & energy should all have the same shape.")
    # make namedtuple
    Data = namedtuple("Data", ["cn1", "cn2", "energy"])
    return Data(cn1=cn1_array, cn2=cn2_array, energy=energy.flatten())


def visualize_cn(irc, rxn_num, aimd=None, disp=None,dial=None,added=None, fpath=None, title=None, xtitle=None,
                 ytitle=None, e_range = None):
    """Plot coordination numbers (CN) of data points for a specific reaction.

    Parameters
    ----------
    irc: namedtuple
        Namedtuple containing 1D arrays of 'cn1', 'cn2', and 'energy' attributes for IRC
        calculations.
    rxn_num: str
        'rxn01'
    aimd: namedtuple, optional
        Namedtuple containing 1D arrays of 'cn1', 'cn2', and 'energy' attributes for AIMD 
        calculations.
    disp: namedtuple, optional
        Namedtuple containing 1D arrays of 'cn1', 'cn2', and 'energy' attributes for normal
        mode displacements.
    dial: namedtuple, optional
        Namedtuple containing 1D arrays of 'cn1', 'cn2', and 'energy' attributes for irc
        dialation.
    added: namedtuple, optional
        Namedtuple containing 1D arrays of 'cn1', 'cn2', and 'energy' attributes for additional 
        data generated in active learning.
    fpath: str, optional
        Path to the output directory; it will be created if it does not exist. If None,
        the plot pops up using `plt.show()`.
    title: str, optional
        Title of the plot.
    xtitle: str, optional
        Title of x-axis corresponding to 'cn2' attribute.
    ytitle: str, optional
        Title of y-axis corresponding to 'cn1' attribute.
    e_range: [min_E,max_E] for setting color range in energy scatter plot

    """
    # general settings
    plt.style.use("seaborn-whitegrid")
    plt.rc("text.latex", preamble=r"\usepackage[cm]{sfmath}")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 18

    # generate plots color coded based on density or energy
    for color_code in ["density", "energy"]:

        # visualization
        fig, ax = plt.subplots(1,1,figsize=(12,8))
        # size = fig.get_size_inches()
        # fig.set_size_inches(size * 2)
        viridis = cm.get_cmap("viridis")

        # plot AIMD and DISP data
        if color_code == "density":
            # calculate point density and use it for coloring
            avail = []
            if aimd is not None:
                avail.extend([aimd.cn1, aimd.cn2])
            if disp is not None:
                avail.extend([disp.cn1, disp.cn2])
            if dial is not None:
                avail.extend([dial.cn1, dial.cn2])
            if added is not None:
                avail.extend([added.cn1, added.cn2])
            # combined = np.vstack(avail)
            #
            avail = np.array(avail)
            # print("Avail: ",avail.shape)
            combined = avail # np.vstack(avail)
            # print(combined.shape)
            # combined = np.array([np.concatenate(avail[:,0]),np.concatenate(avail[:,1])])


            #combined = np.concatenate([data for data in avail],axis=0)
            print(combined.shape)
            try:
                z = gaussian_kde(combined)(combined)
            except:
                continue

            if aimd is not None:
                ax.scatter(aimd.cn2, aimd.cn1, s=20, c=z, marker=".", cmap=viridis)

            if disp is not None:
                ax.scatter(disp.cn2, disp.cn1, s=20, c=z, marker=".", cmap=viridis)

            if dial is not None:
                ax.scatter(dial.cn2, dial.cn1, s=20, c=z, marker="X", cmap=viridis)
                
            if added is not None:
                ax.scatter(added.cn2, added.cn1, s=20, c=z, marker="X", cmap=viridis)

        elif color_code == "energy":
            # if aimd is not None and disp is not None:
            #     raise ValueError("Plot energy for AIMD & DISP together is not implemented!")
            bw_cutoff = -26.744 * rxn_dict[rxn_num]['natoms']
            all_cn1 = []
            all_cn2 = []
            energies = []

            if aimd is not None:
                all_cn1.append(aimd.cn1)
                all_cn2.append(aimd.cn2)
                energies.append(aimd.energy)
            if disp is not None:
                all_cn1.append(disp.cn1)
                all_cn2.append(disp.cn2)
                energies.append(disp.energy)
            if dial is not None:
                all_cn1.append(dial.cn1)
                all_cn2.append(dial.cn2)
                energies.append(dial.energy)
            if added is not None:
                all_cn1.append(added.cn1)
                all_cn2.append(added.cn2)
                energies.append(added.energy)
            cn1 = np.concatenate(all_cn1,axis=0)
            cn2 = np.concatenate(all_cn2, axis=0)
            energies = np.concatenate(energies, axis=0)
            if e_range == None:
                min_E = np.min(energies)
                max_E = bw_cutoff
            else:
                min_E,max_E = e_range
            viridis.set_over("red")
            scat = ax.scatter(cn2, cn1, s=60, c=energies, marker=".", cmap=viridis,
                              vmin=min_E, vmax=max_E)

            cbar = fig.colorbar(scat, ax=ax)
            cbar.set_label(r"$\Delta$ E / kcal mol$^{-1}$", rotation=270, labelpad=20)

            # if disp is not None:
            #
            #     scat = ax.scatter(disp.cn2, disp.cn1, s=5, c=disp.energy, marker="X", cmap=viridis,
            #                       vmin=min(disp.energy), vmax=max(irc.energy) + 10.0)

        else:
            raise ValueError(f"color_ode={color_code} not recognized!")

        # plot IRC pathway
        ax.plot(irc.cn2, irc.cn1, c="black", markersize=4, marker="o", linestyle="dashed",
                label="IRC", fillstyle="full")

        # mark IRC's reactant (R), transition state (TS), & product (P) and display them on top
        ax.plot(irc.cn2[0], irc.cn1[0], c="red", markersize=10, marker="^", linestyle="",
                label="R", zorder=19)
        ax.plot(irc.cn2[irc.energy.argmax()], irc.cn1[irc.energy.argmax()], c="red", markersize=15,
                linestyle="", marker="*", label="TS", zorder=20)
        ax.plot(irc.cn2[-1], irc.cn1[-1], c="red", markersize=10, marker="v", linestyle="",
                label="P", zorder=18)

        # add axis labels, title, and legend
        if xtitle is not None:
            plt.xlabel(xtitle)
        if ytitle is not None:
            plt.ylabel(ytitle)
        if title is not None:
            plt.title(title)
        ax.legend(loc="upper right", fontsize="large", frameon=True)
        plt.tight_layout()

        # save or show figure
        if fpath is not None:
            print(fpath)
            fig.savefig(os.path.join(fpath, f"cn_{color_code}.eps"))
            fig.savefig(os.path.join(fpath, f"cn_{color_code}.png"))
            plt.close(fig)
        else:
            plt.show()
            plt.close()
