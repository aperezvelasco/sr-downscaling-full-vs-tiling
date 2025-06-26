import xarray as xr
import xskillscore as xs
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

"""
Script to generate Table 2.

Performance comparison of Swin2SR for the patch-based approach across seasonal and
annual metrics. This table presents the RMSE, MAE, Bias, SSIM, and PSNR values for the
Swin2SR in the full-domain approach (Swin2SR-F) and the same model structure in the 
patches approach both with (Swin2SR-P) and without overlap (Swin2SR-T), evaluated 
across different seasons (DJF, JJA, MAM, SON) and annually.
"""

# Paths and config
methods_dict = {
    "Swin2SR F.": "/home/pereza/data/phd/project/super-resolution/predictions/full-domain/swin2sr/predictions.nc",
    "Swin2SR T.": "/home/pereza/data/phd/project/super-resolution/predictions/tiles/weighted-sampling/swin2sr-t/predictions.nc",
    "Swin2SR P.": "/home/pereza/data/phd/project/super-resolution/predictions/tiles/weighted-sampling/swin2sr-p/predictions.nc",
}
target_path = "/home/pereza/data/phd/project/super-resolution/data/cerra/t2m/None/*.nc"
output_csv_path = "/tmp/02.csv"
variable = "t2m"
group_by = ["overall", "season"]

# Load datasets
methods = {name: xr.open_dataset(path)[variable] for name, path in methods_dict.items()}
target = xr.open_mfdataset(target_path)[variable]

# Align spatial and temporal domain
lat_min = max(ds.lat.min() for ds in methods.values())
lat_max = min(ds.lat.max() for ds in methods.values())
lon_min = max(ds.lon.min() for ds in methods.values())
lon_max = min(ds.lon.max() for ds in methods.values())
time_min = max(ds.time.min() for ds in methods.values())
time_max = min(ds.time.max() for ds in methods.values())

for name in methods:
    methods[name] = (
        methods[name]
        .sel(
            time=slice(time_min, time_max),
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        )
        .load()
    )

target = target.sel(
    time=slice(time_min, time_max),
    lat=slice(lat_min, lat_max),
    lon=slice(lon_min, lon_max),
).load()

# Grouping logic
group_funcs = {
    "month": lambda x: x.dt.month,
    "season": lambda x: x.dt.season,
}
if group_by is None:
    group_by = ["overall"]


# SSIM and PSNR functions
def compute_ssim(da_pred, da_true):
    return np.mean(
        [
            ssim(
                da_true.isel(time=i).fillna(0),
                da_pred.isel(time=i).fillna(0),
                data_range=float(
                    da_pred.isel(time=i).max() - da_pred.isel(time=i).min()
                ),
            )
            for i in range(len(da_pred.time))
        ]
    )


def compute_psnr(da_pred, da_true):
    return np.mean(
        [
            psnr(
                da_true.isel(time=i).fillna(0),
                da_pred.isel(time=i).fillna(0),
                data_range=float(
                    da_pred.isel(time=i).max() - da_pred.isel(time=i).min()
                ),
            )
            for i in range(len(da_pred.time))
        ]
    )


results = []


def append_metrics(group_label, method_name, pred, truth):
    results.append(
        {
            "Group": group_label,
            "Method": method_name,
            "RMSE": round(float(xs.rmse(pred, truth, dim="time").mean()), 4),
            "MSE": round(float(xs.mse(pred, truth, dim="time").mean()), 4),
            "MAE": round(float(xs.mae(pred, truth, dim="time").mean()), 4),
            "Bias": round(float(xs.me(pred, truth, dim="time").mean()), 4),
            "SSIM": round(compute_ssim(pred, truth), 4),
            "PSNR": round(compute_psnr(pred, truth), 4),
        }
    )


for group in group_by:
    if group == "overall":
        for name, pred in methods.items():
            append_metrics("Year", name, pred, target)
    else:
        func = group_funcs[group]
        target_grouped = target.groupby(func(target.time))
        for name, pred in methods.items():
            pred_grouped = pred.groupby(func(pred.time))
            for grp_label in pred_grouped.groups:
                append_metrics(
                    grp_label, name, pred_grouped[grp_label], target_grouped[grp_label]
                )

# Create DataFrame and export
df = pd.DataFrame(results)
month_map = {
    1: "M01:January",
    2: "M02:February",
    3: "M03:March",
    4: "M04:April",
    5: "M05:May",
    6: "M06:June",
    7: "M07:July",
    8: "M08:August",
    9: "M09:September",
    10: "M10:October",
    11: "M11:November",
    12: "M12:December",
}
season_map = {
    "DJF": "Season:DJF",
    "MAM": "Season:MAM",
    "JJA": "Season:JJA",
    "SON": "Season:SON",
}
df["Group"] = df["Group"].replace(month_map).replace(season_map)
df = df.sort_values(by=["Method", "Group"])
df.set_index(["Group", "Method"]).unstack().round(2).transpose().to_csv(output_csv_path)

print("Evaluation complete. Results saved to:", output_csv_path)
