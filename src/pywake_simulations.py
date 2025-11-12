import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.colors import Normalize
from matplotlib import cm

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# pywake and surrogate_interface modules
from py_wake.deficit_models import ZongGaussianDeficit
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.flow_map import HorizontalGrid
from py_wake.site._site import UniformSite
from py_wake.site.shear import PowerShear
from py_wake.turbulence_models import CrespoHernandez
from py_wake.utils.plotting import setup_plot
from surrogates_interface.surrogates import TensorFlowModel

# wind_farm_loads modules
from iea3_4_pywake_openfast_1 import iea3_4
from wind_farm_loads.py_wake import (
    PropagateDownwindNoSelfInduction, get_rotor_averaged_wind_speed_and_turbulence_intensity
)
from wind_farm_loads.tool_agnostic import compute_sector_average, make_polar_grid, plot_flow_map
from wind_farm_loads.py_wake import compute_flow_map
from wind_farm_loads.tool_agnostic import plot_sector_average


def get_turbine_positions(rotor_diameter, spacing, wd):
    """
    Compute the 2D positions (X, Y) of three wind turbines in a symmetric linear layout 
    based on specified spacing and wind direction.

    This layout calculation was used in the Technical University of Munich's (TUM) 
    control-oriented load surrogate studies, specifically for simulations involving 
    the IEA 3.35 MW reference wind turbine.

    Parameters
    ----------
    rotor_diameter : float
        Rotor diameter of the wind turbines in meters. For the IEA 3.35 MW turbine, 
        this value remains constant across simulations.
    spacing : float
        Turbine spacing in terms of rotor diameters (e.g., 5 means 5D spacing).
    wd : float
        Wind direction in degrees. The layout is rotated according to this direction 
        to reflect the orientation of the turbine array with respect to the inflow.

    Returns
    -------
    tuple of list of floats
        A tuple containing:
            - X coordinates of the three turbines [X_WT1, X_WT2, X_WT3]
            - Y coordinates of the three turbines [Y_WT1, Y_WT2, Y_WT3]

        WT2 is always located at the origin (0, 0). WT1 and WT3 are symmetrically placed 
        on either side along the wind-perpendicular axis, with the layout rotated 
        according to the given wind direction.
    """
    X_WT1 = -spacing * rotor_diameter * np.cos(np.deg2rad(-wd))
    Y_WT1 = spacing * rotor_diameter * np.sin(np.deg2rad(-wd))
    X_WT3 = spacing * rotor_diameter * np.cos(np.deg2rad(-wd))
    Y_WT3 = -spacing * rotor_diameter * np.sin(np.deg2rad(-wd))
    return [X_WT1, 0, X_WT3], [Y_WT1, 0, Y_WT3]

def plot_turb_map(fm,  cmap='Oranges'):
    D = iea3_4.diameter()
    fm.plot(fm.TI_eff, clabel="Added turbulence intensity [-]", levels=100, cmap=cmap, normalize_with=D)
    setup_plot(grid=False, ylabel="Crosswind distance [y/D]", xlabel="Downwind distance [x/D]",
               xlim=[fm.x.min() / D, fm.x.max() / D], ylim=[fm.y.min() / D, fm.y.max() / D], axis='auto')

def pywake_simulations(req_df):

    turbine = iea3_4
    wdir = float(req_df.wd.iloc[0]) # wind direction from FAST.Farm simulation database -8 to 8 degrees
    spacing = req_df.spacing.iloc[0]  # spacing in D
    x,y = get_turbine_positions(rotor_diameter=turbine.diameter(), spacing=spacing, wd=wdir)
    site = UniformSite(ws=req_df.wind_speed.iloc[0],
    ti=req_df.ti.iloc[0]/100,
    shear=PowerShear(h_ref=turbine.hub_height(), alpha=req_df.vertical_shear.iloc[0]),)#type: ignore
    wake_model = PropagateDownwindNoSelfInduction(
    site=site,
    windTurbines=iea3_4,
    deflectionModel=JimenezWakeDeflection(),
    wake_deficitModel=ZongGaussianDeficit(),
    turbulenceModel=CrespoHernandez(),
    )

    # Extract values 
    wd = 270 + wdir # to match FAST.Farm layout and conventions
    ws = req_df.wind_speed.iloc[0]
    yaw = req_df.yaw.to_numpy(dtype=float)
    power_demand = req_df.power_demand.to_numpy()


    sim_res = wake_model(
    x,
    y,
    wd=wd,
    ws=ws,
    yaw=yaw,
    tilt=[0,0,0],
    power_demand=power_demand,
    )
    # get the sector quantities 
    sa = compute_sector_average(sim_res, 
                            radius=9, 
                            n_azimuth_per_sector = 9,
                            look="downwind")
    saws = sa.sel(quantity="WS_eff").values.reshape(3,4)
    sati = sa.sel(quantity="TI_eff").values.reshape(3,4)

    # raws
    raws = sim_res["WS_eff"].values.reshape(3,1)#type: ignore
    rati = sim_res["TI_eff"].values.reshape(3,1)#type: ignore

    x_min, x_max = x[0] - 3*turbine.diameter(), x[2] + 3*turbine.diameter()
    y_min, y_max = -3*turbine.diameter(), 3*turbine.diameter()
    grid = HorizontalGrid(
        x=np.linspace(x_min, x_max, 220),
        y=np.linspace(y_min, y_max, 160),
    )

    return saws, sati, raws, rati


if __name__ == "__main__":

    # load the FAST.Farm results 
    ff_df = pd.read_csv("../ff_database/fast_farm_database.csv")

    all_results = []
    for sim_case in ff_df.simulation_case.unique():
        req_df = ff_df[ff_df.simulation_case == sim_case]
        spacing = req_df.spacing.to_numpy()
        wind_speed = req_df.wind_speed.to_numpy()
        if sim_case % 10 == 0:
            print(f"Processing simulation case: {sim_case}")
        ff = req_df[["saws_u", "saws_r", "saws_d", "saws_l",
                    "sati_u", "sati_r", "sati_d", "sati_l",
                    "raws", "rati"]].to_numpy()
        py_saws, py_sati, py_raws, py_rati = pywake_simulations(req_df)

        # all have 3 rows corresponding to 3 turbines
        py_saws_u = py_saws[:, 0].ravel()
        py_saws_r = py_saws[:, 1].ravel()
        py_saws_d = py_saws[:, 2].ravel()
        py_saws_l = py_saws[:, 3].ravel()

        py_sati_u = (py_sati[:, 0] * 100).ravel()
        py_sati_r = (py_sati[:, 1] * 100).ravel()
        py_sati_d = (py_sati[:, 2] * 100).ravel()
        py_sati_l = (py_sati[:, 3] * 100).ravel()


        ff_saws_u = ff[:, 0].ravel()
        ff_saws_r = ff[:, 1].ravel()
        ff_saws_d = ff[:, 2].ravel()
        ff_saws_l = ff[:, 3].ravel()
        ff_sati_u = ff[:, 4].ravel()
        ff_sati_r = ff[:, 5].ravel()
        ff_sati_d = ff[:, 6].ravel()
        ff_sati_l = ff[:, 7].ravel()
        ff_raws = ff[:, 8].ravel()
        ff_rati = ff[:, 9].ravel()

        case_df = pd.DataFrame({
            "simulation_case": [sim_case] * 3,
            "py_saws_u": py_saws_u,
            "py_saws_r": py_saws_r,
            "py_saws_d": py_saws_d,
            "py_saws_l": py_saws_l,
            "py_sati_u": py_sati_u,
            "py_sati_r": py_sati_r,
            "py_sati_d": py_sati_d,
            "py_sati_l": py_sati_l,
            "py_raws" : py_raws.ravel(),
            "py_rati" : (py_rati * 100).ravel(),   
            "ff_saws_u": ff_saws_u,
            "ff_saws_r": ff_saws_r, 
            "ff_saws_d": ff_saws_d,
            "ff_saws_l": ff_saws_l,
            "ff_sati_u": ff_sati_u,
            "ff_sati_r": ff_sati_r,
            "ff_sati_d": ff_sati_d,
            "ff_sati_l": ff_sati_l,
            "ff_raws" : ff_raws,
            "ff_rati" : ff_rati,
            "spacing" : spacing,
            "wind_speed": wind_speed.ravel(),
            "ti": req_df.ti.to_numpy().ravel(),
            "vertical_shear": req_df.vertical_shear.to_numpy().ravel(),
            "yaw": req_df.yaw.to_numpy().ravel(),
            "power_demand": req_df.power_demand.to_numpy().ravel()
        })

        # add to list
        all_results.append(case_df)

    # concatenate all cases into one dataframe
    final_df = pd.concat(all_results, ignore_index=True)

    print(final_df.columns)
    # save to csv
    os.makedirs("../pywake_simulations", exist_ok=True)
    final_df.to_csv("../pywake_simulations/pywake_simulations.csv", index=False)