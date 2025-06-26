import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

"""
Script to generate Figure 3.

Visual representation of the data division into patches in the tiling implementation. 
On the left side, ERA5 orography data with an example of two different 
CERRA patches (red) surrounded by their corresponding 
ERA5 patch (blue) of size (13, 13). On the right side, CERRA orography with the 
full domain divided into 40 equal-size tiles.

Author: Antonio Pérez Velasco
License: MIT
Last modified: 2025-06-26
"""

# ------------------------------------------------------------------------------
# Load input data (CERRA and ERA5 orography)
# ------------------------------------------------------------------------------

cerra_file_path = "orog_None_cerra_198501_0p05deg.nc"
era5_file_path = "orog_None_era5_198501_0p25deg.nc"

cerra_ds = xr.open_dataset(cerra_file_path)
era5_ds = xr.open_dataset(era5_file_path)

cerra_orog = cerra_ds["orog"].where(cerra_ds["orog"] >= 0, 0)
era5_orog = era5_ds["orog"].where(era5_ds["orog"] >= 0, 0)

# ------------------------------------------------------------------------------
# Domain definition and patching setup
# ------------------------------------------------------------------------------

cerra_lon_min, cerra_lon_max = -10, 6
cerra_lat_min, cerra_lat_max = 35, 45
patch_size = 40

res_cerra = float(cerra_ds.lon[1] - cerra_ds.lon[0])
res_era5 = float(era5_ds.lon[1] - era5_ds.lon[0])
upscale_ratio = res_era5 / res_cerra


# ------------------------------------------------------------------------------
# Utility function
# ------------------------------------------------------------------------------


def sample_surrounding_patch(dataset, target_patch):
    """
    Given a high-resolution patch (target_patch), sample a larger surrounding area
    from the low-resolution dataset to include some context.

    Parameters:
        dataset (xarray.DataArray): The low-res dataset to sample from (e.g. ERA5).
        target_patch (xarray.DataArray): The high-res patch (e.g. CERRA).

    Returns:
        xarray.DataArray: Low-res patch surrounding the target high-res patch.
    """
    gap = (res_era5 / 2 - res_cerra / 2) - 1e-3
    lon_pad = res_era5 * 2
    lat_pad = res_era5 * 2

    lon_min = round(float(target_patch.lon.min().values), 2)
    lon_max = round(float(target_patch.lon.max().values), 2)
    lat_min = round(float(target_patch.lat.min().values), 2)
    lat_max = round(float(target_patch.lat.max().values), 2)

    lat_slice = slice(lat_min - gap - lat_pad, lat_max + gap + lat_pad)
    lon_slice = slice(lon_min - gap - lon_pad, lon_max + gap + lon_pad)

    return dataset.sel(lat=lat_slice, lon=lon_slice)


# ------------------------------------------------------------------------------
# Colormap and color normalization
# ------------------------------------------------------------------------------

base_cmap = plt.get_cmap("terrain")
cmap = mcolors.LinearSegmentedColormap.from_list(
    "extended_terrain", base_cmap(np.linspace(0.4, 1, 256))
)
norm = mcolors.Normalize(vmin=0, vmax=3000)

# ------------------------------------------------------------------------------
# Figure setup and background plotting
# ------------------------------------------------------------------------------

fig, axs = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 6),
    subplot_kw={"projection": ccrs.PlateCarree()},
    gridspec_kw={"wspace": 0},
)

era5_orog.plot(
    ax=axs[0],
    x="lon",
    y="lat",
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
)
cerra_orog.plot(
    ax=axs[1],
    x="lon",
    y="lat",
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
)

for ax in axs:
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.set_extent(
        [cerra_lon_min - 1, cerra_lon_max + 1, cerra_lat_min - 1, cerra_lat_max + 1],
        crs=ccrs.PlateCarree(),
    )

# ------------------------------------------------------------------------------
# Draw patches (full domain on right, two samples on left)
# ------------------------------------------------------------------------------

lat_start_idx = int((cerra_lat_min - cerra_ds.lat.min().item()) / res_cerra)
lon_start_idx = int((cerra_lon_min - cerra_ds.lon.min().item()) / res_cerra)

patch_count = 0

for i in range(
    lat_start_idx,
    lat_start_idx + int((cerra_lat_max - cerra_lat_min) / res_cerra),
    patch_size,
):
    for j in range(
        lon_start_idx,
        lon_start_idx + int((cerra_lon_max - cerra_lon_min) / res_cerra),
        patch_size,
    ):

        cerra_patch = cerra_orog.isel(
            lat=slice(i, i + patch_size), lon=slice(j, j + patch_size)
        )
        era5_patch = sample_surrounding_patch(era5_orog, cerra_patch)

        # Right subplot: all CERRA patches
        rect_right = mpatches.Rectangle(
            (float(cerra_patch.lon.min()), float(cerra_patch.lat.min())),
            cerra_patch.shape[1] * res_cerra,
            cerra_patch.shape[0] * res_cerra,
            fill=False,
            edgecolor="red",
            linewidth=1,
        )
        axs[1].add_patch(rect_right)

        # Left subplot: only two examples
        if patch_count < 2:
            linestyle = "-" if patch_count == 0 else "--"

            # CERRA tile on left subplot
            rect_left = mpatches.Rectangle(
                (float(cerra_patch.lon.min()), float(cerra_patch.lat.min())),
                cerra_patch.shape[1] * res_cerra,
                cerra_patch.shape[0] * res_cerra,
                fill=False,
                edgecolor="red",
                linewidth=1,
                linestyle=linestyle,
            )
            axs[0].add_patch(rect_left)

            # Corresponding ERA5 patch
            era5_rect = mpatches.Rectangle(
                (float(era5_patch.lon.min()), float(era5_patch.lat.min())),
                era5_patch.shape[1] * res_era5,
                era5_patch.shape[0] * res_era5,
                fill=False,
                edgecolor="blue",
                linewidth=1,
                linestyle=linestyle,
            )
            axs[0].add_patch(era5_rect)

        patch_count += 1

# ------------------------------------------------------------------------------
# Colorbar and save figure
# ------------------------------------------------------------------------------

cbar_ax = fig.add_axes([0.05, 0.08, 0.9, 0.05])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label="Elevation (m)")

plt.tight_layout()
plt.savefig("/tmp/03.png", bbox_inches="tight")
