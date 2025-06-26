import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

"""
Script to generate Figure 12.

Difference between ERA5 and CERRA temperature anomalies, computed as the mean 
temperature for 2015–2020 (validation and testing periods) minus the mean for 1985–2014
(training period). Positive values indicate areas where ERA5 shows a stronger anomaly 
than CERRA. The orange rectangle highlights the region analyzed in Figure 13 
(latitudes 41.60–42.20◦N and longitudes 1.60–2.40◦E).

Author: Antonio Pérez Velasco
Date: 2025-06-26
"""

# -----------------------
# Filepaths (editable)
# -----------------------
era5_path = "/home/pereza/data/phd/project/super-resolution/data/era5/t2m/None/*.nc"
cerra_path = "/home/pereza/data/phd/project/super-resolution/data/cerra/t2m/None/*.nc"
output_path = "/tmp/16.png"

# -----------------------
# Load datasets
# -----------------------
ds_era5 = xr.open_mfdataset(era5_path)
ds_cerra = xr.open_mfdataset(cerra_path).sel(lat=slice(35, 45), lon=slice(-10, 6))


# -----------------------
# Harmonize domain and time range
# -----------------------
def compute_common_bounds(datasets):
    """Compute the spatial and temporal intersection across datasets."""
    lat_min = max(ds.lat.min().item() for ds in datasets.values())
    lat_max = min(ds.lat.max().item() for ds in datasets.values())
    lon_min = max(ds.lon.min().item() for ds in datasets.values())
    lon_max = min(ds.lon.max().item() for ds in datasets.values())
    time_min = max(ds.time.min().item() for ds in datasets.values())
    time_max = min(ds.time.max().item() for ds in datasets.values())
    return lat_min, lat_max, lon_min, lon_max, time_min, time_max


lat_min, lat_max, lon_min, lon_max, time_min, time_max = compute_common_bounds(
    {"ERA5": ds_era5, "CERRA": ds_cerra}
)

# Apply spatial and temporal cropping
ds_era5 = ds_era5.sel(
    time=slice(time_min, time_max),
    lat=slice(lat_min, lat_max),
    lon=slice(lon_min, lon_max),
)
ds_cerra = ds_cerra.sel(
    time=slice(time_min, time_max),
    lat=slice(lat_min, lat_max),
    lon=slice(lon_min, lon_max),
)

# Match coordinates and time exactly
ds_era5 = ds_era5.sel(time=ds_cerra.time)
ds_era5 = ds_era5.interp(lat=ds_cerra.lat, lon=ds_cerra.lon, method="nearest")


# -----------------------
# Anomaly calculation
# -----------------------
def compute_anomaly(
    ds,
    var="t2m",
    base_period=("1985-01-01", "2014-12-31"),
    recent_period=("2015-01-01", "2020-12-31"),
):
    """Compute anomaly as the difference in mean temperature between two periods."""
    baseline = ds[var].sel(time=slice(*base_period)).mean("time")
    recent = ds[var].sel(time=slice(*recent_period)).mean("time")
    return recent - baseline


era5_anomaly = compute_anomaly(ds_era5)
cerra_anomaly = compute_anomaly(ds_cerra)
anomaly_diff = era5_anomaly - cerra_anomaly

# -----------------------
# Plot
# -----------------------
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Main map
im = anomaly_diff.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    vmin=-1.5,
    vmax=1.5,
    levels=11,
    extend="both",
    add_colorbar=False,
)

# Colorbar
cbar = plt.colorbar(im, orientation="horizontal", pad=0.04, shrink=0.8, ax=ax)
cbar.set_label("Temperature anomaly difference (ºC)", fontsize=12)

# Add features
ax.coastlines(linewidths=1)
ax.add_feature(cfeature.BORDERS, linewidth=1)
ax.set_title("")

# Highlighted regions
highlight_boxes = [
    {"lat": (41.6, 42.2), "lon": (1.6, 2.4), "color": "orange"},
    {"lat": (38.0, 38.6), "lon": (-4.4, -3.7), "color": "green"},
]

for box in highlight_boxes:
    rect = mpatches.Rectangle(
        (box["lon"][0], box["lat"][0]),
        box["lon"][1] - box["lon"][0],
        box["lat"][1] - box["lat"][0],
        linewidth=2,
        edgecolor=box["color"],
        facecolor="none",
        transform=ccrs.PlateCarree(),
        zorder=10,
    )
    ax.add_patch(rect)

# Final layout
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
