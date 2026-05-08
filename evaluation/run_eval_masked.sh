#!/usr/bin/env bash
# ============================================================================
# Project:     Accurate and Robust Localization of Landmarks on a Vehicle
# Author:      Adam Běhoun <xbehoua00@vutbr.cz>
# Year:        2026
# Description: Evaluates relative pose estimation performance of feature extractors by computing AUC of pose error curves.
# ============================================================================

for folder in 2024_04_11_16_33_46 2024_04_12_08_22_26 2024_04_12_11_49_59 2024_04_11_16_42_57 2024_04_12_10_12_39 2024_04_11_17_08_10; do
    echo "======================================"
    echo "Processing Scene: $folder"
    echo "======================================"

    output_dir=/zfs-pool/home/xbehoua00/imc/evaluation/$folder

    python imc-pose-evaluation.py \
        --image-dir /zfs-pool/home/xbehoua00/imc/dataset/$folder/masked/ \
        --gt-dir /zfs-pool/home/xbehoua00/imc/rec-sift/$folder/sift/sfm_best/ \
        --output-json "$output_dir/metrics.json" \
        --plot-prefix "$folder"
done
echo "ALL SCENES COMPLETED."
