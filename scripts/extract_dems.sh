#!/bin/bash
# Extract downloaded DEM TIF files to NPY format and install
# them into assets/Terrains/SouthPole/.
# Requires: gdal (gdalinfo), python3, numpy, pyyaml

echo "Extracting DEMs..."

CWD=$(pwd)
DEMS_PATH="$CWD/tmp"
TGT_PATH="$CWD/assets/Terrains/SouthPole"

echo "Reading DEM info..."
for dem in "$DEMS_PATH"/*.tif; do
    dem_name_no_ext="${dem%.*}"
    gdalinfo "$dem" > "$dem_name_no_ext.info"
    python3 scripts/process_info.py --info_path "$dem_name_no_ext.info" --output_dir "$DEMS_PATH" --output_name "$(basename "$dem_name_no_ext")"
    python3 scripts/preprocess_dem.py --dem_path "$dem" --output_dir "$DEMS_PATH" --output_name "$(basename "$dem_name_no_ext").npy"
done
echo "Finished extracting DEMs."

echo "Moving DEMs to $TGT_PATH..."
mkdir -p "$TGT_PATH"
for dem in "$DEMS_PATH"/*.npy; do
    dem_name_no_ext="${dem%.*}"
    dem_name_no_ext=$(basename "$dem_name_no_ext")
    mkdir -p "$TGT_PATH/$dem_name_no_ext"
    mv "$dem" "$TGT_PATH/$dem_name_no_ext/dem.npy"
    mv "$DEMS_PATH/$dem_name_no_ext.yaml" "$TGT_PATH/$dem_name_no_ext/dem.yaml"
done
echo "Finished moving DEMs."

echo "Cleaning up tmp/..."
rm -rf "$DEMS_PATH"
echo "Done."
