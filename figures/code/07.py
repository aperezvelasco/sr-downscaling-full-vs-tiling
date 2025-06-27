import numpy as np
import xarray as xr
import xskillscore
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

"""
Script to generate Figure 7.

Spatial evaluation results for the different SR downscaling methods for the full-domain
approach, with the methods represented in columns, Bicubic interpolation 
(simple benchmark), UNet (30M parameters), DeepESD (300M parameters) and Swin2SR 
(30M parameters), from left to right, and evaluation metrics in rows 
(bias and RMSE, from top to bottom).

Author: Antonio Pérez Velasco
Last modified: 2025-06-26
"""

# ------------------------------------------------------------------------------
# Input paths and settings
# ------------------------------------------------------------------------------

main_dir = "/home/pereza/data/phd/project/super-resolution/predictions/full-domain"
target_path = "/home/pereza/data/phd/project/super-resolution/data/cerra/t2m/None"
output_path = "/tmp/07.png"
filename = "predictions.nc"
varname = "t2m"
metrics_to_compute = ["Bias", "RMSE"]
spatial_domain = None

predictions_paths = {
    "Bicubic Interp.": f"{main_dir}/bicubic",
    "UNet": f"{main_dir}/unet2d",
    "DeepESD": f"{main_dir}/deepesd",
    "Swin2SR": f"{main_dir}/swin2sr",
}

# ------------------------------------------------------------------------------
# Load predictions and target
# ------------------------------------------------------------------------------

predictions = {
    name: xr.open_dataset(f"{path}/{filename}")[varname]
    for name, path in predictions_paths.items()
}
target = xr.open_mfdataset(f"{target_path}/*.nc")[varname]

# Ensure overlapping domain (time and space)
lat_min = max([ds.lat.min() for ds in predictions.values()] + [target.lat.min()])
lat_max = min([ds.lat.max() for ds in predictions.values()] + [target.lat.max()])
lon_min = max([ds.lon.min() for ds in predictions.values()] + [target.lon.min()])
lon_max = min([ds.lon.max() for ds in predictions.values()] + [target.lon.max()])
time_min = max([ds.time.min() for ds in predictions.values()] + [target.time.min()])
time_max = min([ds.time.max() for ds in predictions.values()] + [target.time.max()])

for name in predictions:
    predictions[name] = (
        predictions[name]
        .sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
            time=slice(time_min, time_max),
        )
        .load()
    )

target = target.sel(
    lat=slice(lat_min, lat_max),
    lon=slice(lon_min, lon_max),
    time=slice(time_min, time_max),
).load()

# ------------------------------------------------------------------------------
# Compute metrics
# ------------------------------------------------------------------------------

metrics = {metric: {} for metric in metrics_to_compute}
for name, pred in predictions.items():
    if "Bias" in metrics:
        metrics["Bias"][name] = xskillscore.me(pred, target, dim="time")
    if "RMSE" in metrics:
        metrics["RMSE"][name] = xskillscore.rmse(pred, target, dim="time")

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------

fig, axs = plt.subplots(
    nrows=len(metrics),
    ncols=len(predictions),
    figsize=(len(predictions) * 10, len(metrics) * 12),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

div_cmap = "bwr"
seq_cmap = "OrRd"

vmin_vmax = {
    "Bias": (-1.2, 1.2),
    "RMSE": (0.6, 2.4),
}

for i, metric in enumerate(metrics):
    vmin, vmax = vmin_vmax[metric]
    cmap = div_cmap if vmin < 0 and vmax > 0 else seq_cmap

    if cmap == div_cmap:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
        levels = 13
    else:
        levels = 11

    for j, name in enumerate(metrics[metric]):
        ax = axs[i, j]
        data = metrics[metric][name]

        mappable = data.plot(
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            add_colorbar=False,
            extend="both",
        )
        mean_val = float(data.mean().values)
        ax.set_title(f"{name} (Mean: {mean_val:.2f})", fontsize=42)
        ax.coastlines(linewidths=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        if spatial_domain is not None:
            ax.set_extent(
                [spatial_domain[2], spatial_domain[3], spatial_domain[0], spatial_domain[1]],
                crs=ccrs.PlateCarree(),
            )

    # Add colorbar
    cbar_ax = fig.add_axes(
        [
            0.01,
            0.07
            + (len(metrics) - i - 1) * (1 / len(metrics))
            - (0.07 * (len(metrics) - i - 1)),
            0.98,
            0.05,
        ]
    )
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks(np.linspace(vmin, vmax, levels)[::2])
    cbar_ax.tick_params(labelsize=32)
    cbar_ax.set_title(metric, fontsize=34)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
