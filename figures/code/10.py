import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

"""
Script to generate Figure 10.

Spatial maps of the target (CERRA) and predictions from different super-resolution
methods for specific weather events in the Iberian Peninsula. Columns correspond to the 
dates of January 12th, February 3rd, and 4th, 2020, capturing periods of cold and warm 
anomalies. Rows represent results from bicubic interpolation, UNet, DeepESD, SwinSR, 
and SwinSR (Patches).

Author: Antonio Pérez Velasco
Last modified: 2025-06-26
"""

# ------------------------------------------------------------------------------
# Load datasets for different methods and the CERRA target
# ------------------------------------------------------------------------------

main_dir = "/home/pereza/data/phd/project/super-resolution/predictions"
filename = "predictions.nc"

ds_prediction_1 = xr.open_dataset(f"{main_dir}/full-domain/bicubic/{filename}")["t2m"]
ds_prediction_2 = xr.open_dataset(f"{main_dir}/full-domain/unet2d/{filename}")["t2m"]
ds_prediction_3 = xr.open_dataset(f"{main_dir}/full-domain/deepesd/{filename}")["t2m"]
ds_prediction_4 = xr.open_dataset(f"{main_dir}/full-domain/swin2sr/{filename}")["t2m"]
ds_prediction_5 = xr.open_dataset(
    f"{main_dir}/tiles/weighted-sampling/swin2sr-p/{filename}"
)["t2m"]

ds_target = xr.open_mfdataset(
    "/home/pereza/data/phd/project/super-resolution/data/cerra/t2m/None/*.nc"
)["t2m"]

# ------------------------------------------------------------------------------
# Harmonize spatial extent
# ------------------------------------------------------------------------------

lat_min = max(
    [
        float(ds.lat.min())
        for ds in [
            ds_prediction_1,
            ds_prediction_2,
            ds_prediction_3,
            ds_prediction_4,
            ds_prediction_5,
            ds_target,
        ]
    ]
)
lat_max = min(
    [
        float(ds.lat.max())
        for ds in [
            ds_prediction_1,
            ds_prediction_2,
            ds_prediction_3,
            ds_prediction_4,
            ds_prediction_5,
            ds_target,
        ]
    ]
)
lon_min = max(
    [
        float(ds.lon.min())
        for ds in [
            ds_prediction_1,
            ds_prediction_2,
            ds_prediction_3,
            ds_prediction_4,
            ds_prediction_5,
            ds_target,
        ]
    ]
)
lon_max = min(
    [
        float(ds.lon.max())
        for ds in [
            ds_prediction_1,
            ds_prediction_2,
            ds_prediction_3,
            ds_prediction_4,
            ds_prediction_5,
            ds_target,
        ]
    ]
)

datasets = [
    ds_prediction_1,
    ds_prediction_2,
    ds_prediction_3,
    ds_prediction_4,
    ds_prediction_5,
    ds_target,
]

dates_to_plot = ["2020-01-12T12:00:00", "2020-02-03T12:00:00", "2020-02-04T12:00:00"]

for i in range(len(datasets)):
    datasets[i] = (
        datasets[i].sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        ).sel(
            time=dates_to_plot
        ).load()
    )

(
    ds_prediction_1,
    ds_prediction_2,
    ds_prediction_3,
    ds_prediction_4,
    ds_prediction_5,
    ds_target,
) = datasets

# ------------------------------------------------------------------------------
# Define SR methods and target
# ------------------------------------------------------------------------------

methods = {
    "Bicubic interp.": ds_prediction_1,
    "UNet": ds_prediction_2,
    "DeepESD": ds_prediction_3,
    "Swin2SR": ds_prediction_4,
    "Swin2SR (Patches)": ds_prediction_5,
}

# ------------------------------------------------------------------------------
# Plotting setup
# ------------------------------------------------------------------------------

fig, axs = plt.subplots(
    len(methods) + 1,
    len(dates_to_plot),
    figsize=(20, 25),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

# Plot CERRA (target)
for j, date in enumerate(dates_to_plot):
    ax = axs[0, j]
    target_data = ds_target.sel(time=date)
    mappable = target_data.plot(ax=ax, cmap="bwr", vmin=-3, vmax=25, add_colorbar=False)
    ax.set_title(f"Target: {date[:10]} 12:00", fontsize=24)
    ax.coastlines()

# Plot predictions
for i, (method_name, method_data) in enumerate(methods.items(), 1):
    for j, date in enumerate(dates_to_plot):
        ax = axs[i, j]
        prediction_data = method_data.sel(time=date)
        mappable = prediction_data.plot(
            ax=ax, cmap="bwr", vmin=-3, vmax=25, levels=29, add_colorbar=False
        )
        ax.set_title(f"{method_name}: {date[:10]} 12:00", fontsize=22)
        ax.coastlines()

# ------------------------------------------------------------------------------
# Add colorbar and save figure
# ------------------------------------------------------------------------------

plt.subplots_adjust(hspace=0, wspace=0, bottom=0.15)
cbar_ax = fig.add_axes([0.02, -0.05, 0.96, 0.03])
cbar_ax.tick_params(labelsize=26)
cbar_ax.set_title("Temperature (ºC)", fontsize=26)

cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
cbar.set_ticks(np.linspace(-3, 25, 29)[::2])

plt.tight_layout()
plt.savefig("/tmp/10.png",bbox_inches="tight")
