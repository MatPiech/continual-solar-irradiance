#!/bin/bash

# Generate clear-sky data for multiple locations using clear_sky.py

# Create output directory if it doesn't exist
for location in folsom modena poznan karpicko alpnach bern1 bern2 neuchatel; do
    mkdir -p "data/${location}_prepared"
done

echo "============================================================"
echo "Generating clear-sky data for multiple locations..."
echo "============================================================"

echo -e "\n[1/8] Folsom, USA"
python scripts/clear_sky.py \
    --latitude 38.642 \
    --longitude -121.148 \
    --timezone "US/Pacific" \
    --start "2016-01-01 00:00:00" \
    --end "2016-12-31 23:59:00" \
    --output-path "data/folsom_prepared/folsom_clear_sky.csv"

echo -e "\n[2/8] Modena, Italy"
python scripts/clear_sky.py \
    --latitude 44.6471 \
    --longitude 10.9252 \
    --timezone "Europe/Rome" \
    --start "2024-06-01 00:00:00" \
    --end "2024-06-30 23:59:00" \
    --output-path "data/modena_prepared/modena_clear_sky.csv"

echo -e "\n[3/8] Poznan, Poland"
python scripts/clear_sky.py \
    --latitude 52.4064 \
    --longitude 16.9252 \
    --timezone "Europe/Warsaw" \
    --start "2024-10-18 00:00:00" \
    --end "2025-07-12 23:59:00" \
    --output-path "data/poznan_prepared/poznan_clear_sky.csv"

echo -e "\n[4/8] Karpicko, Poland"
python scripts/clear_sky.py \
    --latitude 52.7500 \
    --longitude 16.3333 \
    --timezone "Europe/Warsaw" \
    --start "2024-09-01 00:00:00" \
    --end "2025-11-08 23:59:00" \
    --output-path "data/karpicko_prepared/karpicko_clear_sky.csv"

echo -e "\n[5/8] Alpnach, Switzerland"
python scripts/clear_sky.py \
    --latitude 46.935881 \
    --longitude 8.27584 \
    --timezone "Europe/Zurich" \
    --start "2018-01-01 00:00:00" \
    --end "2018-12-31 23:59:00" \
    --output-path "data/alpnach_prepared/alpnach_clear_sky.csv"

echo -e "\n[6/8] Bern1, Switzerland"
python scripts/clear_sky.py \
    --latitude 46.953617 \
    --longitude 7.42305 \
    --timezone "Europe/Zurich" \
    --start "2018-01-01 00:00:00" \
    --end "2018-06-11 23:59:00" \
    --output-path "data/bern1_prepared/bern1_clear_sky.csv"

echo -e "\n[7/8] Bern2, Switzerland"
python scripts/clear_sky.py \
    --latitude 46.953133 \
    --longitude 7.422567 \
    --timezone "Europe/Zurich" \
    --start "2018-01-01 00:00:00" \
    --end "2018-04-10 23:59:00" \
    --output-path "data/bern2_prepared/bern2_clear_sky.csv"

echo -e "\n[8/8] Neuchatel, Switzerland"
python scripts/clear_sky.py \
    --latitude 46.99785 \
    --longitude 6.947415 \
    --timezone "Europe/Zurich" \
    --start "2018-01-01 00:00:00" \
    --end "2018-12-31 23:59:00" \
    --output-path "data/neuchatel_prepared/neuchatel_clear_sky.csv"
