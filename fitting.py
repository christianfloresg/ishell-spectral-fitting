import numpy as np
from numpy.typing import NDArray
from spectra import ProplydData, MoogStokesModel, BTSettlModel
import matplotlib.pyplot as plt

def chi_squared(ydata: NDArray, yerrdata: NDArray, ymodel: NDArray) -> float:
    """Calculate the chi-squared statistic for the given data and model spectrum.
    All arrays must be the same length.
    """
    return np.nansum((ydata-ymodel)**2/yerrdata**2)

def compute_moogstokes_chi2_grid(data: ProplydData, Teff_vals: NDArray, logg_vals:
                                 NDArray, rK_vals: NDArray, vsini_vals: NDArray,
                                 B_vals: NDArray,renormalization: NDArray | None = None,
                                 shifts: NDArray | None = None, regions: list[int] | None = None) -> NDArray:
    """Computes the chi-squared statistic across a grid of MoogStokes models.
    The models are interpolated to match the x-values of the data.

    Bear in mind that this function can become excruciatingly slow if the grid
    of parameters is too large.

    Parameters
    ----------
    data: ProplydData
        Science data to fit.
    Teff_vals: NDArray
        Effective temperatures across the grid.
    ...
    regions: list of int or None, optional
        Wavelength regions of MoogStokes models to fit. If None, all regions (0
        through 6) will be used.
    
    renormalization: list of floats, optional
        Small changes to the normalization values that can be applied to each region
    shifts: NDArray, optional
        Shifts in pixels applied to each region. Due to innacuracies in wavelength position
        of lines, and wavelength calibration of data we expect some pixel level shifts.
    Returns
    -------
    chi2: NDArray
        Chi-squared statistic across the grid of parameters indexed in the order
        (Teff, logg, rK, vsini, B).
    """


    if regions is None:
        regions = range(7)

    if shifts is None:
        shifts = np.zeros(len(regions))

    if renormalization is None:
        renormalization = np.ones(len(regions))

    chi2 = np.zeros( (len(Teff_vals), len(logg_vals), len(rK_vals), len(vsini_vals), len(B_vals)) )
    chi2_region = np.zeros( (len(Teff_vals), len(logg_vals), len(rK_vals), len(vsini_vals), len(B_vals)) )

    for nn, r in enumerate(regions):

        data.doppler_shift_data(shifts[nn])

        xlo, xhi = MoogStokesModel.region_xlims(r)
        xdata, ydata, yerrdata = data.get_range(xlo, xhi)

        #### Apply the desired normalization
        ydata, yerrdata = ydata*renormalization[nn] , yerrdata*renormalization[nn]

        xdata = np.array(xdata)
        ydata = np.array(ydata)
        yerrdata = np.array(yerrdata)

        for i, Teff in enumerate(Teff_vals):
            for j, logg in enumerate(logg_vals):
                for k, rK in enumerate(rK_vals):
                    for l, vsini in enumerate(vsini_vals):
                        for m, B in enumerate(B_vals):

                            model = MoogStokesModel(Teff, logg, rK, B, vsini, r)
                            ymodel = model.interpolate(xdata)
                            ymodel = np.array(ymodel)

                            chi2_region[i, j, k, l, m] = chi_squared(ydata, yerrdata, ymodel)

        ### sum chi_squared of different regions
        chi2 = chi2 + chi2_region
        chi2_region = np.zeros((len(Teff_vals), len(logg_vals), len(rK_vals), len(vsini_vals), len(B_vals)))

        ### get spectrum back to original position so another shift can be applied to next region

        data.doppler_shift_data(-shifts[nn])
        print(renormalization[nn])
    return chi2

def best_chi2_grid_params(chi2: NDArray, Teff_vals: NDArray, logg_vals: NDArray,
                          rK_vals: NDArray, vsini_vals: NDArray, B_vals: NDArray) -> tuple[float, float, float, float, float]:
    """Returns the best-fit parameters (minimum chi-squared) from a grid generated
    by compute_moogstokes_chi2_grid().

    Parameters
    ----------
    chi2: NDArray
        Chi-squared statistic across the grid of parameters indexed in the order
        (Teff, logg, rK, vsini, B).
    """
    idxs = np.unravel_index(chi2.argmin(), chi2.shape)
    Teff_best = Teff_vals[idxs[0]]
    logg_best = logg_vals[idxs[1]]
    rK_best = rK_vals[idxs[2]]
    vsini_best = vsini_vals[idxs[3]]
    B_best = B_vals[idxs[4]]
    
    return Teff_best, logg_best, rK_best, vsini_best, B_best


def plot_chi2_slice(
        chi2_grid: np.ndarray,
        param1_vals: np.ndarray,
        param2_vals: np.ndarray,
        param1_name: str,
        param2_name: str,
        other_axes: dict = None,
        param_labels: dict = None
):
    """
    Plot the 2D chi-squared grid for any pair of parameters by minimizing over the other axes.

    other_axes: dict, e.g. { 'rK': idx_rK, 'vsini': idx_vsini, 'B': idx_B }
        If specified, slices those axes at the given indices instead of minimizing.
    param_labels: dict, label overrides for pretty plot titles.
    """
    if param_labels is None:
        param_labels = {}
    # Map from names to axis indices in the grid
    param_axes = ['Teff', 'logg', 'rK', 'vsini', 'B']
    param_idxs = {k: i for i, k in enumerate(param_axes)}

    idx1 = param_idxs[param1_name]
    idx2 = param_idxs[param2_name]

    # Axes to minimize/slice over
    axes = list(range(chi2_grid.ndim))
    axes.remove(idx1)
    axes.remove(idx2)
    min_axes = axes
    # If you want to slice (fix) some axes instead of minimizing, you can do so
    if other_axes:
        # Build slices
        slicer = [slice(None)] * chi2_grid.ndim
        for k, v in other_axes.items():
            slicer[param_idxs[k]] = v
        reduced_chi2 = chi2_grid[tuple(slicer)]
    else:
        # Profile/minimize over other parameters
        reduced_chi2 = np.min(chi2_grid, axis=tuple(min_axes))

    # Make the plot
    plt.figure(figsize=(7, 6))
    CS = plt.contourf(param1_vals, param2_vals, reduced_chi2.T, levels=30, cmap='viridis')
    plt.colorbar(CS, label='Chi-squared')
    plt.xlabel(param_labels.get(param1_name, param1_name))
    plt.ylabel(param_labels.get(param2_name, param2_name))
    plt.title(
        f'Chi-squared: {param_labels.get(param1_name, param1_name)} vs {param_labels.get(param2_name, param2_name)}')
    plt.scatter(
        param1_vals[np.argmin(np.min(reduced_chi2, axis=1))],
        param2_vals[np.argmin(np.min(reduced_chi2, axis=0))],
        marker='*', color='red', s=120, label='Minimum'
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_chi2_confidence_interval(param_vals, chi2_grid, axis, delta_chi2=1.0):
    """
    Computes confidence interval for a single parameter (profiling over the rest).

    param_vals: np.ndarray, grid values for the parameter
    chi2_grid: np.ndarray, chi2 grid
    axis: int, axis of param_vals in chi2_grid
    delta_chi2: float, e.g. 1.0 for 1-sigma, 4.72 for 4 parameters (see table!)
    """
    # Profile over all other parameters (min over the rest)
    chi2_profile = np.min(chi2_grid, axis=tuple(a for a in range(chi2_grid.ndim) if a != axis))
    min_chi2 = np.min(chi2_profile)
    ok = (chi2_profile - min_chi2) <= delta_chi2
    # Return parameter range (can be more than one interval if grid is gappy)
    if np.any(ok):
        return param_vals[ok].min(), param_vals[ok].max()
    else:
        return None, None


def get_all_confidence_intervals(chi2_grid, param_grids, delta_chi2=1.0):
    """
    Returns dict of confidence intervals for each parameter.
    param_grids: dict, e.g., {'Teff': Teff_vals, ...}
    """
    results = {}
    for i, (name, vals) in enumerate(param_grids.items()):
        lo, hi = get_chi2_confidence_interval(vals, chi2_grid, i, delta_chi2=delta_chi2)
        results[name] = (lo, hi)
    return results

def plot_1d_chi2_profile(param_vals, chi2_grid, axis, param_name, delta_chi2=1.0):
    # Profile over all other parameters
    chi2_profile = np.min(chi2_grid, axis=tuple(a for a in range(chi2_grid.ndim) if a != axis))
    min_chi2 = np.min(chi2_profile)
    threshold = min_chi2 + delta_chi2

    plt.figure(figsize=(6,4))
    plt.plot(param_vals, chi2_profile, label=f'χ² profile: {param_name}')
    plt.axhline(threshold, color='orange', linestyle='--', label=f'χ² min + {delta_chi2}')
    plt.axvline(param_vals[np.argmin(chi2_profile)], color='red', linestyle=':', label='Best fit')
    plt.xlabel(param_name)
    plt.ylabel('Chi-squared')
    plt.title(f'Profile χ² for {param_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Report confidence interval
    ok = (chi2_profile - min_chi2) <= delta_chi2
    if np.any(ok):
        conf_interval = (param_vals[ok].min(), param_vals[ok].max())
    else:
        conf_interval = (None, None)
    print(f'Best fit {param_name}: {param_vals[np.argmin(chi2_profile)]}')
    print(f'{delta_chi2:.2f} delta chi² confidence interval: {conf_interval}')
    return conf_interval


def compute_reduced_chi2_bestfit(
    data,
    best_params,
    regions,
    num_params=5,
    MoogStokesModel=MoogStokesModel
):
    """
    Compute reduced chi-squared for best-fit parameters given data and regions.

    Parameters
    ----------
    data : ProplydData
        The observed data object.
    best_params : tuple
        Best-fit parameters: (Teff, logg, rK, vsini, B)
    regions : list[int]
        List of region indices to stitch.
    num_params : int
        Number of fitted model parameters (default: 5).
    MoogStokesModel : class
        The model class used for generating synthetic spectra.

    Returns
    -------
    reduced_chi2 : float
        The reduced chi-squared value for the best fit.
    """
    Teff_best, logg_best, rK_best, vsini_best, B_best = best_params

    xdata = []
    ydata = []
    yerrdata = []
    ymodel = []

    for r in regions:
        xlo, xhi = MoogStokesModel.region_xlims(r)
        xtemp, ytemp, yerrtemp = data.get_range(xlo, xhi)
        model = MoogStokesModel(Teff_best, logg_best, rK_best, B_best, vsini_best, r)
        ymodeltemp = model.interpolate(xtemp)

        xdata += list(xtemp)
        ydata += list(ytemp)
        yerrdata += list(yerrtemp)
        ymodel += list(ymodeltemp)

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    yerrdata = np.array(yerrdata)
    ymodel = np.array(ymodel)

    # Only use finite (non-nan) values for chi2 calculation
    valid = np.isfinite(ydata) & np.isfinite(yerrdata) & np.isfinite(ymodel)
    n_points = np.sum(valid)
    dof = n_points - num_params

    chi2 = np.nansum((ydata[valid] - ymodel[valid]) ** 2 / yerrdata[valid] ** 2)
    reduced_chi2 = chi2 / dof

    return reduced_chi2


def plot_bestfit_and_residuals(
        data,
        best_params,
        regions,
        MoogStokesModel=MoogStokesModel,
        shifts=None,
        renormalization=None,
        obj_name="object",
        outdir="figures/model_plots",
        ylim=(0.55, 1.1),
):
    """
    Plot the data, best-fit model, and residuals (with errors) for each region.

    Parameters
    ----------
    data : ProplydData
        Observed data object.
    best_params : tuple
        (Teff, logg, rK, vsini, B)
    regions : list[int]
        List of region indices.
    MoogStokesModel : class
        Model class (default: MoogStokesModel).
    obj_name : str
        Name of object for figure title/output.
    outdir : str
        Output directory for saving the figure.
    ylim : tuple
        y-limits for spectra panels.
    """
    Teff_best, logg_best, rK_best, vsini_best, B_best = best_params


    n_regions = len(regions)
    nrows = n_regions

    if shifts is None:
        shifts = np.zeros(n_regions)

    if renormalization is None:
        renormalization = np.ones(n_regions)

    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 2.2 * nrows), gridspec_kw={"width_ratios": [3, 2]})

    for idx, r in enumerate(regions):
        # Get data in this region

        shift=shifts[idx]
        data.doppler_shift_data(shift)

        xlo, xhi = MoogStokesModel.region_xlims(r)
        xdata, ydata, yerrdata = data.get_range(xlo, xhi)
        ydata,yerrdata = ydata*renormalization[idx],yerrdata*renormalization[idx]
        model = MoogStokesModel(Teff_best, logg_best, rK_best, B_best, vsini_best, r)
        ymodel = model.interpolate(xdata)

        # Spectra panel
        ax = axs[idx, 0]
        ax.plot(xdata, ydata, c='midnightblue', label='Data')
        ax.plot(xdata, ymodel, c='magenta', label='Model')
        ax.fill_between(xdata, ydata - yerrdata, ydata + yerrdata, color='blue', alpha=0.2, lw=0)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(*ylim)
        ax.set_ylabel("Flux")
        if idx == 0:
            ax.legend(fontsize=9)
        ax.set_title(f"Region {r}")
        if idx == nrows - 1:
            ax.set_xlabel("Wavelength (Å)")

        # Residuals panel
        ax_res = axs[idx, 1]
        residual = ydata - ymodel
        ax_res.axhline(0, color='gray', lw=1)
        ax_res.plot(xdata, residual, c='teal', label='Residual')
        ax_res.fill_between(xdata, -yerrdata, yerrdata, color='orange', alpha=0.2, lw=0, label='±1σ error')
        ax_res.set_xlim(xlo, xhi)
        ax_res.set_ylabel("Residuals")
        # Visual scale: residuals, error
        ax_res.set_ylim(-4.5 * np.nanmedian(yerrdata), 4.5 * np.nanmedian(yerrdata))
        if idx == 0:
            ax_res.legend(fontsize=9)
        if idx == nrows - 1:
            ax_res.set_xlabel("Wavelength (Å)")

        data.doppler_shift_data(-shift)

    fig.suptitle(
        f"{obj_name}: Teff={Teff_best}, logg={logg_best}, rK={round(rK_best,2)}, vsini={vsini_best}, B={B_best}",
        fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # Save if outdir is provided
    import os
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(
        f"{outdir}/{obj_name}_Regions{regions}_Teff{Teff_best}_logg{logg_best}_rK{round(rK_best,2)}_vsini{vsini_best}_B{B_best}.png",
        dpi=120, facecolor="white")
    plt.show()


def automatic_wavelength_shifts_values(data: ProplydData, Teff: float, logg:
                                float, rK: float, vsini: float,
                                 B: float, guess_shift: int, regions: list[int] | None = None) -> NDArray:

    """
    find the best pixel shifts for a specified number of regions for a given model.
    The values of the model are not too important as the lines detected and observed are there.
    :param data: ProplydData
        Observed data object.
    :param Teff: 
    :param logg: 
    :param rK: 
    :param vsini: 
    :param B: 
    :param guess_shift: guess shift to run the chi2 minimzation 
    :param regions: 
    :return: 
    """

    if regions is None:
        regions = range(7)


    best_shift=np.empty(len(regions))

    shift_array=range(-30+guess_shift,30+guess_shift)
    chi2 = np.zeros( (len(regions), len(shift_array)) )

    for nn, r in enumerate(regions):

        for ii, shifts in enumerate(shift_array):

            data.doppler_shift_data(shifts)

            xlo, xhi = MoogStokesModel.region_xlims(r)
            xdata, ydata, yerrdata = data.get_range(xlo, xhi)

            xdata = np.array(xdata)
            ydata = np.array(ydata)
            yerrdata = np.array(yerrdata)

            model = MoogStokesModel(Teff, logg, rK, B, vsini, r)
            ymodel = model.interpolate(xdata)
            ymodel = np.array(ymodel)

            chi2[nn, ii] = chi_squared(ydata, yerrdata, ymodel)

            ### get spectrum back to original position so another shift can be applied to next region
            data.doppler_shift_data(-shifts)


        this_region_min_chi2=np.nanmin(chi2[nn,:])
        best_shift[nn]=shift_array[int(np.nanargmin(chi2[nn,:]))]+guess_shift
        # print(this_region_min_chi2)
        # print(best_shift[nn])

    return best_shift
