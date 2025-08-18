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
                                 shifts: NDArray | None = None, regions: list[int] | None = None,
                                 resolution: float | None = None, kernel: str | None = None ) -> NDArray:
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

        ####### THIS BLOCK NEEDS TO BE COMMENTED OUT ######

        rng = np.random.default_rng(42)
        noise = rng.normal(loc=0.0, scale=1 / 100., size=len(ydata))
        ydata = ydata + noise
        mean_snr = np.nanmean(yerrdata)
        yerrdata = yerrdata * 100 ** (-1) / mean_snr

        print(np.nanmean(yerrdata))

        for i, Teff in enumerate(Teff_vals):
            for j, logg in enumerate(logg_vals):
                for k, rK in enumerate(rK_vals):
                    for l, vsini in enumerate(vsini_vals):
                        for m, B in enumerate(B_vals):

                            model = MoogStokesModel(Teff, logg, rK, B, vsini, r)
                            if resolution is not None or kernel is not  None:
                                model.resolution_change(resolution=resolution, Kernel=kernel)

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
