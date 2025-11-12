
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from pywake_simulations import pywake_simulations
from iea3_4_pywake_openfast_1 import iea3_4


import pandas as pd


def compare_sectors(sim_case, ff_df):
    # compare ff and pywake sectors 
    req_df = ff_df[ff_df.simulation_case == sim_case]

    ws = req_df[["saws_u", "saws_r", "saws_d", "saws_l"]]
    ti = req_df[["sati_u", "sati_r", "sati_d", "sati_l"]]
    fig, axs = plt.subplots(2, 6, figsize=(20, 5))

    # --- FAST.Farm sectors ---
    for i in range(3):
        values = ws.to_numpy()[i]
        norm = Normalize(vmin=np.min(ws.to_numpy()), vmax=np.max(ws.to_numpy()))
        equal_values = [1] * len(values)
        colors = [cm.Blues(norm(v)) for v in values]  # light → dark blue

        wedges, _ = axs[0][i].pie(equal_values, colors=colors, startangle=135,
                                  counterclock=False, wedgeprops={'edgecolor': 'black'})
        for k, wedge in enumerate(wedges):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = 0.6 * np.cos(np.deg2rad(angle))
            y = 0.6 * np.sin(np.deg2rad(angle))
            axs[0][i].text(x, y, f"{values[k]:.2f}", ha='center', va='center',
                           fontsize=9, fontweight='bold')

        axs[0][i].set_title(f"Turbine {i+1}", fontsize=12, fontweight='bold')

    for i in range(3, 6):
        idx = i - 3
        values = ti.to_numpy()[idx]
        norm = Normalize(vmin=np.min(ti.to_numpy()), vmax=np.max(ti.to_numpy()))
        equal_values = [1] * len(values)
        colors = [cm.Oranges(norm(v)) for v in values]  # light → dark orange

        wedges, _ = axs[0][i].pie(equal_values, colors=colors, startangle=135,
                                  counterclock=False, wedgeprops={'edgecolor': 'black'})
        for k, wedge in enumerate(wedges):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = 0.6 * np.cos(np.deg2rad(angle))
            y = 0.6 * np.sin(np.deg2rad(angle))
            axs[0][i].text(x, y, f"{values[k]:.2f}", ha='center', va='center',
                           fontsize=9, fontweight='bold')

        axs[0][i].set_title(f"Turbine {idx+1}", fontsize=12, fontweight='bold')

    # --- PyWake sectors ---
    saws, sati, raws, rati = pywake_simulations(req_df)

    for i in range(3):
        values = saws[i]
        norm = Normalize(vmin=np.min(saws), vmax=np.max(saws))
        equal_values = [1] * len(values)
        colors = [cm.Blues(norm(v)) for v in values]

        wedges, _ = axs[1][i].pie(equal_values, colors=colors, startangle=135,
                                  counterclock=False, wedgeprops={'edgecolor': 'black'})
        for k, wedge in enumerate(wedges):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = 0.6 * np.cos(np.deg2rad(angle))
            y = 0.6 * np.sin(np.deg2rad(angle))
            axs[1][i].text(x, y, f"{values[k]:.2f}", ha='center', va='center',
                           fontsize=9, fontweight='bold')

        axs[1][i].set_title("PyWake WS", fontsize=12, fontweight='bold')

    for i in range(3, 6):
        idx = i - 3
        values = sati[idx] * 100
        norm = Normalize(vmin=np.min(sati) * 100, vmax=np.max(sati) * 100)
        equal_values = [1] * len(values)
        colors = [cm.Oranges(norm(v)) for v in values]

        wedges, _ = axs[1][i].pie(equal_values, colors=colors, startangle=135,
                                  counterclock=False, wedgeprops={'edgecolor': 'black'})
        for k, wedge in enumerate(wedges):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = 0.6 * np.cos(np.deg2rad(angle))
            y = 0.6 * np.sin(np.deg2rad(angle))
            axs[1][i].text(x, y, f"{values[k]:.2f}", ha='center', va='center',
                           fontsize=9, fontweight='bold')

        axs[1][i].set_title("PyWake TI", fontsize=12, fontweight='bold')
    plt.tight_layout()
    compare_ra(raws, rati, req_df)

def compare_ra(raws, rati, req_df):
    # compare ff and pywake ra
    ff_raws = req_df[["raws"]].to_numpy().reshape(3,1)
    ff_rati = req_df[["rati"]].to_numpy().reshape(3,1)
    raws = raws.flatten()
    rati = rati.flatten()*100
    ff_raws = ff_raws.flatten()
    ff_rati = ff_rati.flatten()

    # Number of turbines
    turbines = np.arange(1, 4)

    # Bar width and positions
    bar_width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]  # relative to turbine index

    plt.figure(figsize=(8, 5))

    # Wind Speed (blue shades)
    plt.bar(turbines + offsets[0]*bar_width, raws, width=bar_width, color="#1f77b4", label="PyWake WS")
    plt.bar(turbines + offsets[1]*bar_width, ff_raws, width=bar_width, color="#8ecae6", label="FAST.Farm WS")

    # Turbulence Intensity (orange shades)
    plt.bar(turbines + offsets[2]*bar_width, rati, width=bar_width, color="#ff7f0e", label="PyWake TI")
    plt.bar(turbines + offsets[3]*bar_width, ff_rati, width=bar_width, color="#ffd8a8", label="FAST.Farm TI")

    # Labels and aesthetics
    plt.xticks(turbines, [f"T{i}" for i in turbines])
    plt.xlabel("Turbine", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Comparison of Rotor-Averaged Inflows (PyWake vs FAST.Farm)", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # load the FAST.Farm results 
    ff_df = pd.read_csv("../ff_database/fast_farm_database.csv")

    compare_sectors(sim_case=38, ff_df=ff_df)
