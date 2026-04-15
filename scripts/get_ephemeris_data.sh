#!/bin/bash
# Download JPL NAIF kernel files for celestial body positioning.
# These files are required by the StellarEngine for accurate
# sun/earth/moon ephemeris calculations.

echo "Downloading ephemeris data..."

CWD=$(pwd)
mkdir -p assets/Ephemeris
cd assets/Ephemeris

wget -q --no-check-certificate https://naif.jpl.nasa.gov/pub/naif/JUNO/kernels/spk/de421.bsp
wget -q --no-check-certificate https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_080317.tf
wget -q --no-check-certificate https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/a_old_versions/pck00008.tpc
wget -q --no-check-certificate https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de421_1900-2050.bpc

cd "$CWD"
echo "Ephemeris data downloaded to assets/Ephemeris/"
