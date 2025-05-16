import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import rasterio.plot
import rasterio.features
from rasterio.mask import mask as raster_mask
from rasterio.plot import show


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

    fig, ax = plt.subplots(figsize=(16, 16))

    # Grid
    with rasterio.open("data/constraints/grid_10km.tif") as src:
        original_transform = src.transform
        grid_data, grid_transform = raster_mask(src, mask_geom, crop=False, indexes=2)
    masked_grid = np.ma.masked_where(grid_data == -9999, grid_data)
    masked_grid = np.ma.masked_where(masked_grid > distance_grid, masked_grid)
    show(masked_grid, transform=grid_transform, ax=ax, cmap="terrain", title="Cropped Grid", origin='lower')

    # Streets
    with rasterio.open("data/constraints/street_10km.tif") as src:
        street_data, street_transform = raster_mask(src, mask_geom, crop=False, indexes=2)
    masked_street = np.ma.masked_where(street_data == -9999, street_data)
    masked_street = np.ma.masked_where(masked_street > distance_street, masked_street)
    show(masked_street, transform=street_transform, ax=ax, cmap="terrain", title="Cropped Street", origin='lower')

    # Energy production potential
    with rasterio.open("data/constraints/final_5.1.tif") as src:
        epp_data, epp_transform = raster_mask(src, mask_geom, crop=False, indexes=1)
    masked_epp = np.ma.masked_where(epp_data == -9999, epp_data)
    masked_epp = np.ma.masked_where(masked_epp < energy_production, masked_epp)
    show(masked_epp, transform=epp_transform, ax=ax, cmap="terrain", title="Cropped Energy Production Potential", origin='lower')





    # BLN
    bln = gpd.read_file("data/constraints/bundesinventare-bln_2056.shp")
    memfile = rasterize_vector(bln, original_transform)
    with memfile.open() as dataset:
        cropped_bln, bln_transform = rasterio.mask.mask(dataset, mask_geom, crop=False, indexes=1)
    cropped_bln = np.ma.masked_where(cropped_bln == 0, cropped_bln)
    #show(cropped_bln, transform=bln_transform, ax=ax, cmap="terrain", title="Cropped BLN", origin='lower')

    # Parks
    parks = gpd.read_file("data/constraints/paerke/N2025_Revision_Park_20241016.shp")
    memfile = rasterize_vector(parks, original_transform)
    with memfile.open() as dataset:
        cropped_parks, parks_transform = rasterio.mask.mask(dataset, mask_geom, crop=False, indexes=1)
    cropped_parks = np.ma.masked_where(cropped_parks == 0, cropped_parks)
    #show(cropped_parks, transform=parks_transform, ax=ax, cmap="terrain", title="Cropped Parks", origin='lower')

    # DHM 25
    def extract_avg_z(geom):
        coords = list(geom.coords)
        z_vals = [z for x, y, z in coords]
        return sum(z_vals) / len(z_vals)
    dhm25 = gpd.read_file("data/constraints/DHM25_BM_SHP/dhm25_l").to_crs(epsg=2056)
    dhm25['elevation'] = dhm25.geometry.apply(extract_avg_z)
    dhm25 = dhm25[dhm25['elevation'] >= elevation]
    memfile = rasterize_vector(dhm25, original_transform)
    with memfile.open() as dataset:
        cropped_dhm25, dhm25_transform = rasterio.mask.mask(dataset, mask_geom, crop=False, indexes=1)
    cropped_dhm25 = np.ma.masked_where(cropped_dhm25 == 0, cropped_dhm25)
    show(cropped_dhm25, transform=dhm25_transform, ax=ax, cmap="terrain", title="Cropped DHM25", origin='lower')

    plt.show()

    intersection_mask = [masked_grid, masked_street, cropped_dhm25]#masked_epp, masked_dhm25]
    exclusion_mask = [cropped_bln, cropped_parks]

    min_rows_in = min([masked.shape[0] for masked in intersection_mask])
    min_cols_in = min([masked.shape[1] for masked in intersection_mask])

    min_rows_ex = min([masked.shape[0] for masked in exclusion_mask])
    min_cols_ex = min([masked.shape[1] for masked in exclusion_mask])

    min_rows = min(min_rows_in, min_rows_ex)
    min_cols = min(min_cols_in, min_cols_ex)

    masked_epp = masked_epp[:min_rows, :min_cols]
    masked_grid = masked_grid[:min_rows, :min_cols]
    masked_street = masked_street[:min_rows, :min_cols]
    cropped_bln = cropped_bln[:min_rows, :min_cols]
    cropped_parks = cropped_parks[:min_rows, :min_cols]
    cropped_dhm25 = cropped_dhm25[:min_rows, :min_cols]

    final_mask = (
            ~masked_epp.mask &
            ~masked_grid.mask &
            ~masked_street.mask &
            ~cropped_dhm25.mask &
            cropped_bln.mask &
            cropped_parks.mask
    )

    final_masked = np.ma.masked_where(~final_mask, np.ones_like(final_mask, dtype=np.uint8))
    plt.imshow(final_masked, cmap="terrain", origin="lower")
    plt.title("Final Suitability Mask")
    plt.show()

    return final_masked, original_transform


def rasterize_vector(geo_df, transform=None, resolution=100):
    from rasterio.io import MemoryFile
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    bounds = geo_df.total_bounds
    minx, miny, maxx, maxy = bounds
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

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


if __name__ == "__main__":
    main()