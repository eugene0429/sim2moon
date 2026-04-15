#!/bin/bash
# Download LOLA 5m-per-pixel DEM tiles from NASA PGDA.
# These are used by the LandscapeBuilder for realistic
# South Pole background terrain.

echo "Downloading DEM files..."
echo "Using 8 parallel processes. This may take a while..."

CWD=$(pwd)
mkdir -p tmp
cd tmp

cat "$CWD/scripts/dems_list.txt" | xargs -n 1 -P 8 wget -nv -nc

cd "$CWD"

echo "Finished downloading DEMs to tmp/"
echo "Run ./scripts/extract_dems.sh to process and install them."
