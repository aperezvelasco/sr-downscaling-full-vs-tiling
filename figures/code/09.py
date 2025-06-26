import numpy as np
import xarray as xr
import xskillscore
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

"""
Script to generate Figure 9.

As Figure 8, but comparing the Swin2SR in the full-domain approach (Swin2SR (F))
with the same model structure for the two patch-based implementations, using 
non-overlapping tiles (Swin2SR (T)), and using overlapping patches (Swin2SR (P)), 
in the second and third columns respectively.

Author: Antonio Pérez Velasco
Last modified: 2025-06-26
"""

# ------------------------------------------------------------------------------
# Input paths and settings
# ------------------------------------------------------------------------------

main_dir = "/home/pereza/data/phd/project/super-resolution/predictions"
target_path = "/home/pereza/data/phd/project/super-resolution/data/cerra/t2m/None"
output_path = "/tmp/09.png"
filename = "predictions.nc"
varname = "t2m"
baseline_method = "Swin2SR (F)"
metrics_to_compute = ["Bias", "RMSE"]

spatial_domain = None

predictions_paths = {
    "Swin2SR (T)": f"{main_dir}/tiles/weighted-sampling/swin2sr-t",
    "Swin2SR (P)": f"{main_dir}/tiles/weighted-sampling/swin2sr-p",
    "Swin2SR (F)": f"{main_dir}/full-domain/swin2sr",
}

# ------------------------------------------------------------------------------
# Load predictions and target
# ------------------------------------------------------------------------------

predictions = {
    name: xr.open_dataset(f"{path}/{filename}")[varname]
    for name, path in predictions_paths.items()
}
target = xr.open_mfdataset(f"{target_path}/*.nc")[varname]

# Ensure overlapping domain (both time and space)
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
# Compute metrics and differences w.r.t. baseline
# ------------------------------------------------------------------------------

metrics = {metric: {} for metric in metrics_to_compute}

for name, pred in predictions.items():
    if "Bias" in metrics:
        metrics["Bias"][name] = xskillscore.me(pred, target, dim="time")
    if "RMSE" in metrics:
        metrics["RMSE"][name] = xskillscore.rmse(pred, target, dim="time")

# Differences
for metric in metrics:
    base = metrics[metric][baseline_method]
    for name in list(metrics[metric].keys()):
        if name != baseline_method:
            diff_name = f"{name} - {baseline_method}"
            metrics[metric][diff_name] = metrics[metric][name] - base
            del metrics[metric][name]

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------

fig, axs = plt.subplots(
    nrows=len(metrics),
    ncols=len(metrics["Bias"]),
    figsize=(len(metrics["Bias"]) * 10, len(metrics) * 12),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

div_cmap = "bwr"
seq_cmap = "OrRd"

vmin_vmax_abs = {
    "Bias": (-1, 1),
    "RMSE": (0.5, 1.5),
}
vmin_vmax_diff = {
    "Bias": (-0.4, 0.4),
    "RMSE": (-0.4, 0.4),
}

for i, metric in enumerate(metrics):
    abs_vmin, abs_vmax = vmin_vmax_abs[metric]
    diff_vmin, diff_vmax = vmin_vmax_diff[metric]

    abs_cmap = div_cmap if abs_vmin < 0 and abs_vmax > 0 else seq_cmap
    diff_cmap = div_cmap if diff_vmin < 0 and diff_vmax > 0 else seq_cmap

    if abs_cmap == div_cmap:
        vmax = max(abs(abs_vmin), abs(abs_vmax))
        abs_vmin, abs_vmax = -vmax, vmax
    if diff_cmap == div_cmap:
        vmax = max(abs(diff_vmin), abs(diff_vmax))
        diff_vmin, diff_vmax = -vmax, vmax

    # Plotting each map
    for j, name in enumerate(metrics[metric]):
        if len(metrics) == 1:
            ax = axs[j]
        else:
            ax = axs[i, j]

        data = metrics[metric][name]

        if "-" not in name:
            mappable_abs = data.plot(
                ax=ax,
                cmap=abs_cmap,
                vmin=abs_vmin,
                vmax=abs_vmax,
                levels=11,
                add_colorbar=False,
                extend="both",
            )
        else:
            mappable_diff = data.plot(
                ax=ax,
                cmap=diff_cmap,
                vmin=diff_vmin,
                vmax=diff_vmax,
                levels=11,
                add_colorbar=False,
                extend="both",
            )
        ax.set_title(name, fontsize=40)
        if spatial_domain is not None:
            ax.set_extent(
                [
                    spatial_domain[2],
                    spatial_domain[3],
                    spatial_domain[0],
                    spatial_domain[1],
                ],
                crs=ccrs.PlateCarree(),
            )
        ax.coastlines(linewidths=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)

    # Add colorbars
    cbar_ax_abs = fig.add_axes(
        [
            0.01,
            0.07
            + (len(metrics) - i - 1) * (1 / len(metrics))
            - (0.07 * (len(metrics) - i - 1)),
            (1 / len(predictions)) - 0.01,
            0.05,
        ]
    )
    cbar_abs = fig.colorbar(mappable_abs, cax=cbar_ax_abs, orientation="horizontal")
    cbar_abs.set_ticks(np.linspace(abs_vmin, abs_vmax, 11)[::3])
    cbar_ax_abs.tick_params(labelsize=34)
    cbar_ax_abs.set_title(metric, fontsize=34)

    cbar_ax_diff = fig.add_axes(
        [
            (1 / len(predictions)) + 0.01,
            0.07
            + (len(metrics) - i - 1) * (1 / len(metrics))
            - (0.07 * (len(metrics) - i - 1)),
            0.98 - (1 / len(predictions)),
            0.05,
        ]
    )
    cbar_diff = fig.colorbar(
        mappable_diff, cax=cbar_ax_diff, orientation="horizontal"
    )
    cbar_diff.set_ticks(np.linspace(diff_vmin, diff_vmax, 11)[::2])
    cbar_ax_diff.tick_params(labelsize=34)
    cbar_ax_diff.set_title(f"Difference in: {metric}", fontsize=34)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
