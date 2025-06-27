import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

"""
Script to generate Figure 1.

Compares the orography of SRTM30+ (top), ERA5 and CERRA (bottom) over the Iberian Peninsula.
Includes common colorbar and overlays marking ERA5 and CERRA spatial domains.

Author: Antonio Pérez Velasco
Last modified: 2025-06-26
"""

# ------------------------------------------------------------------------------
# Input paths and settings
# ------------------------------------------------------------------------------

era5_path = (
    "/home/pereza/data/phd/project/super-resolution/data/era5/orog/None/"
    "orog_None_era5_198501_0p25deg.nc"
)
cerra_path = (
    "/home/pereza/data/phd/project/super-resolution/data/cerra/orog/None/"
    "orog_None_cerra_198501_0p05deg.nc"
)
srtm_path = (
    "/home/pereza/data/phd/project/super-resolution/srtm30plus/orog/None/"
    "orog_None_srtm30plus_198501_0p01deg.nc"
)
output_path = "/tmp/01.png"

lon_min, lon_max = -12.5, 8.5
lat_min, lat_max = 32.5, 47.5
vmin, vmax = 0, 3000

# ------------------------------------------------------------------------------
# Load and preprocess data
# ------------------------------------------------------------------------------

ds_era5 = xr.open_dataset(era5_path)
ds_cerra = xr.open_dataset(cerra_path)
ds_srtm = xr.open_dataset(srtm_path)

orog_era5 = ds_era5["orog"].where(ds_era5["orog"] >= 0, 0)
orog_cerra = ds_cerra["orog"].where(ds_cerra["orog"] >= 0, 0)
orog_srtm = ds_srtm["elev"].where(ds_srtm["elev"] >= 0, 0)

# ------------------------------------------------------------------------------
# Colormap settings
# ------------------------------------------------------------------------------

original_cmap = plt.get_cmap("terrain")
extended_colors = original_cmap(np.linspace(0.4, 1, 256))
cmap = mcolors.LinearSegmentedColormap.from_list("extended_terrain", extended_colors)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------


fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[2, 2], hspace=0.08, wspace=0.02)

# Top row: SRTM (centered using both columns)
srtm_ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
orog_srtm.plot(
    ax=srtm_ax,
    x="lon",
    y="lat",
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
)
srtm_ax.set_title("SRTM Elevation (1 km res.)", fontsize=18)
srtm_ax.set_extent([lon_min, lon_max, lat_min, lat_max])
srtm_ax.add_feature(cfeature.COASTLINE)
srtm_ax.add_feature(cfeature.BORDERS, linestyle=":")

# Bottom-left: ERA5
era5_ax = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
orog_era5.plot(
    ax=era5_ax,
    x="lon",
    y="lat",
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
)
era5_ax.set_title("ERA5 Elevation (25 km res.)", fontsize=18)
era5_ax.set_extent([lon_min, lon_max, lat_min, lat_max])
era5_ax.add_feature(cfeature.COASTLINE)
era5_ax.add_feature(cfeature.BORDERS, linestyle=":")

# Bottom-right: CERRA
cerra_ax = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
orog_cerra.plot(
    ax=cerra_ax,
    x="lon",
    y="lat",
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
)
cerra_ax.set_title("CERRA Elevation (5.5 km res.)", fontsize=18)
cerra_ax.set_extent([lon_min, lon_max, lat_min, lat_max])
cerra_ax.add_feature(cfeature.COASTLINE)
cerra_ax.add_feature(cfeature.BORDERS, linestyle=":")

# ------------------------------------------------------------------------------
# Overlay domain boxes and legends
# ------------------------------------------------------------------------------

for ax_num, ax in enumerate([era5_ax, cerra_ax]):
    ax.add_patch(
        mpatches.Rectangle(
            (-12, 33),
            20,
            14,
            fill=False,
            edgecolor="blue",
            linewidth=2,
            label="ERA5 Domain",
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (-10, 35),
            16,
            10,
            fill=False,
            edgecolor="red",
            linewidth=2,
            label="CERRA Domain",
        )
    )
    if ax_num != 0:
        ax.legend(loc="lower right", fontsize=14)

# ------------------------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------------------------

cbar_ax = fig.add_axes([0.13, 0.05, 0.765, 0.05])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
cbar = fig.colorbar(
    sm,
    cax=cbar_ax,
    orientation="horizontal",
    ticks=np.linspace(vmin, vmax, 7)
)
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Elevation (m)", fontsize=22)

# ------------------------------------------------------------------------------
# Save figure
# ------------------------------------------------------------------------------

plt.savefig(output_path, dpi=300, bbox_inches="tight")
