#!/bin/bash

# Configuration
FORECASTING_HORIZON=15

# Create output directory if it doesn't exist
for location in folsom modena poznan karpicko alpnach bern1 bern2 neuchatel; do
    mkdir -p "data/${location}_prepared"
done

echo -e "\n============================================================"
echo "Calculating Smart Persistence baselines for all locations..."
echo "============================================================"

counter=0
for location in folsom modena poznan karpicko alpnach bern1 bern2 ne; do
    counter=$((counter + 1))
    echo "Processing location [$counter/8]: ${location^^}"
    python scripts/smart_persistence_model.py \
        --periods-path "data/${location}_prepared/periods.pickle" \
        --clear-sky-path "data/${location}_prepared/${location}_clear_sky.csv" \
        --forecasting-horizon $FORECASTING_HORIZON \
	--start-day 0101 \
	--save-results
done

echo -e "\n============================================================"
