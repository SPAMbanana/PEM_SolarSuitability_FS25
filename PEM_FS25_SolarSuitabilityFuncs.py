import geopandas as gpd
import os
import rasterio
from rasterio.mask import mask as raster_mask
import numpy as np
import matplotlib.pyplot as plt
import rasterio.plot
import rasterio.features
from rasterio.features import rasterize


def main():
    print("nothing to see here")


def clip_area(data_path: str, boundaries: str, name: str):
    swissboundaries = gpd.read_file(boundaries)
    area = swissboundaries[swissboundaries['NAME'] == name]
    area = area.to_crs(epsg=2056)

    output_path = "data_output/NCPs_clipped"

    for file in os.listdir(data_path):
        with rasterio.open(f"data/NCPs/{file}") as src:
            profile = src.profile
            mask_geom = [geom.__geo_interface__ for geom in area.geometry]
            masked_data, masked_transform = raster_mask(src, mask_geom, crop=True)
        profile.update({
            "height": masked_data.shape[1],
            "width": masked_data.shape[2],
            "transform": masked_transform
        })
        with rasterio.open(f"{output_path}/masked_{name}_{file}", "w", **profile) as dst:
            dst.write(masked_data)


def generate_mask(gbay_output_path, elevation, energy_production, distance_street, distance_grid, boundaries, name):
    swissboundaries = gpd.read_file(boundaries)
    area = swissboundaries[swissboundaries['NAME'] == name]
    area = area.to_crs(epsg=2056)
    mask_geom = [geom.__geo_interface__ for geom in area.geometry]

    # BLN
    bln = gpd.read_file("data/constraints/bundesinventare-bln_2056.shp")
    memfile = rasterize_vector(bln)
    with memfile.open() as dataset:
        masked_bln, bln_transform = rasterio.mask.mask(dataset, mask_geom, crop=True, indexes=1)
    masked_bln = np.ma.masked_where(masked_bln == 0, masked_bln)

    # Parks
    parks = gpd.read_file("data/constraints/paerke/N2025_Revision_Park_20241016.shp")
    memfile = rasterize_vector(parks)
    with memfile.open() as dataset:
        masked_parks, parks_transform = rasterio.mask.mask(dataset, mask_geom, crop=True, indexes=1)
    masked_parks = np.ma.masked_where(masked_parks == 0, masked_parks)

    # DHM 25
    def extract_avg_z(geom):
        coords = list(geom.coords)
        z_vals = [z for x, y, z in coords]
        return sum(z_vals) / len(z_vals)
    dhm25 = gpd.read_file("data/constraints/DHM25_BM_SHP/dhm25_l").to_crs(epsg=2056)
    dhm25['elevation'] = dhm25.geometry.apply(extract_avg_z)
    dhm25 = dhm25[dhm25['elevation'] >= elevation]
    memfile = rasterize_vector(dhm25)
    with memfile.open() as dataset:
        masked_dhm25, dhm25_transform = rasterio.mask.mask(dataset, mask_geom, crop=True, indexes=1)
    masked_dhm25 = np.ma.masked_where(masked_dhm25 == 0, masked_dhm25)

    # Grid
    with rasterio.open("data/constraints/grid_10km.tif") as src:
        grid_data, masked_transform = raster_mask(src, mask_geom, crop=True, indexes=2)
        grid_crs = src.crs
        grid_transform = src.transform
        grid_profile = src.profile

    masked_grid = np.ma.masked_where(grid_data == -9999, grid_data)
    masked_grid = np.ma.masked_where(masked_grid > distance_grid, masked_grid)

    # Streets
    with rasterio.open("data/constraints/street_10km.tif") as src:
        street_data, masked_transform = raster_mask(src, mask_geom, crop=True, indexes=2)
        street_crs = src.crs
        street_transform = src.transform
        street_profile = src.profile
    masked_street = np.ma.masked_where(street_data == -9999, street_data)
    masked_street = np.ma.masked_where(masked_street > distance_street, masked_street)

    # Energy production potential
    with rasterio.open("data/constraints/final_5.1.tif") as src:
        #epp_data = src.read(1)
        epp_data, masked_transform = raster_mask(src, mask_geom, crop=True, indexes=1)
        epp_crs = src.crs
        epp_transform = src.transform
        epp_profile = src.profile
    masked_epp = np.ma.masked_where(epp_data == -9999, epp_data)
    masked_epp = np.ma.masked_where(masked_epp < energy_production, masked_epp)

    #v_mask = generate_vector_mask(dhm25, bln, parks, area)
    # Plot all masked arrays
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    masked_arrays = [masked_epp, masked_grid, masked_street, masked_bln, masked_parks, masked_dhm25]

    titles = ["Energy Production Potential", "Grid", "Street", "BLN", "Parks", "DHM25"]

    for ax, masked, title in zip(axes, masked_arrays, titles):
        ax.imshow(masked.mask, cmap="terrain", origin="upper")
        #show x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        #ax.axis("off")

    plt.tight_layout()
    plt.show()

    intersection_mask = [masked_grid, masked_street, masked_dhm25]#masked_epp, masked_dhm25]
    exclusion_mask = [masked_bln, masked_parks]

    min_rows_in = min([masked.shape[0] for masked in intersection_mask])
    min_cols_in = min([masked.shape[1] for masked in intersection_mask])

    min_rows_ex = min([masked.shape[0] for masked in exclusion_mask])
    min_cols_ex = min([masked.shape[1] for masked in exclusion_mask])

    min_rows = min(min_rows_in, min_rows_ex)
    min_cols = min(min_cols_in, min_cols_ex)

    cropped_masks_in = [masked[:min_rows, :min_cols].mask for masked in intersection_mask]
    cropped_masks_ex = [masked[:min_rows, :min_cols].mask for masked in exclusion_mask]

    # Combine the masks
    combined_mask_in = np.logical_not(np.any([m for m in cropped_masks_in], axis=0))

    combined_mask_ex = (
            masked_bln[:min_rows, :min_cols].mask &
            masked_parks[:min_rows, :min_cols].mask
    )
    #combined_mask_in = np.logical_or.reduce(cropped_masks_in)
    #combined_mask_ex = np.logical_and.reduce(cropped_masks_ex)

    plt.imshow(combined_mask_in, cmap="terrain", origin="lower")
    plt.title("combined_mask_in")
    plt.show()
    plt.imshow(combined_mask_ex, cmap="terrain", origin="lower")
    plt.title("combined_mask_ex")
    plt.show()
    # Combine the masks
    final_mask = np.logical_and(combined_mask_in, ~combined_mask_ex)

    plt.imshow(final_mask, cmap="terrain", origin="lower")
    plt.title("Final Suitability Mask")
    plt.show()

    return final_mask, masked_transform


def rasterize_vector(geo_df, resolution=25):
    from rasterio.io import MemoryFile
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    bounds = geo_df.total_bounds
    minx, miny, maxx, maxy = bounds
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    shapes = ((geom, 1) for geom in geo_df.geometry)
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": raster.dtype,
        "crs": geo_df.crs,
        "transform": transform
    }

    memfile = MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(raster, 1)

    return memfile


# not used anymore - may be deleted TODO
def generate_vector_mask(base, *args):
    mask = base.copy()
    for gdf in args:
        mask = mask.overlay(gdf, how='difference')
    return mask


def generate_raster_mask(*args):
    # generate mask from raster data
    min_rows = min([masked.shape[0] for masked in args])
    min_cols = min([masked.shape[1] for masked in args])

    cropped_masks = [masked[:min_rows, :min_cols].mask for masked in args]

    combined_mask = np.logical_or.reduce(cropped_masks)
    return combined_mask


def combine_vec_ras(v_mask, r_mask, transform):
    # plot the vector mask and raster mask in the same figure
    fig, ax = plt.subplots(figsize=(12, 6))
    v_mask.plot(ax=ax, color='red', alpha=0.5)
    rasterio.plot.show(r_mask, ax=ax, transform=v_mask.crs.to_proj4(), cmap='gray', title="Energy production potential")
    ax.set_title("Combined Mask")
    plt.tight_layout()
    plt.show()

    from rasterio.features import shapes
    from shapely.geometry import shape

    valid = ~r_mask.mask

    print(valid.any())
    plt.imshow(valid, cmap="terrain", origin="lower")
    shapes_gen = shapes(np.ones_like(valid, dtype='uint8'), mask=valid, transform=transform)

    geoms = [shape(geom) for geom, value in shapes_gen]
    gdf_from_r = gpd.GeoDataFrame(geometry=geoms, crs=r_mask.crs)


"""def rasterize_vector(geo_df):
    bound = geo_df.total_bounds
    minx, miny, maxx, maxy = bound
    width = int((maxx - minx) / 25)
    height = int((maxy - miny) / 25)
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
    shapes = ((geom, 1) for geom in geo_df.geometry)
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    return raster, geo_df.crs, transform"""


if __name__ == "__main__":
    main()