from safe_gpu import safe_gpu

safe_gpu.claim_gpus()

from pathlib import Path
from hloc import match_dense, reconstruction, pairs_from_exhaustive

base_images_dir = Path('/zfs-pool/home/xbehoua00/train_data/masked_images')
base_outputs_dir = Path('/zfs-pool/home/xbehoua00/zms-tool/xbehoua00/train/ALIKED/rec-loftr')

target_folders = ['second', 'fourth', 'sixth']

loftr_conf = match_dense.confs['loftr']  # Use 'loftr_aachen' if preferred

for folder_name in target_folders:
    print(f"\n{'='*40}")
    print(f"Processing folder: {folder_name}")
    print(f"{'='*40}")
    
    images = base_images_dir / folder_name
    
    outputs = base_outputs_dir / folder_name
    outputs.mkdir(parents=True, exist_ok=True)
    
    sfm_pairs = outputs / 'pairs.txt'
    sfm_dir = outputs / 'sfm'
    
    print(f"Generating exhaustive pairs for '{folder_name}'...")
    
    if not images.exists():
        print(f"Warning: Directory {images} does not exist. Skipping...")
        continue

    image_list = [p.relative_to(images).as_posix() for p in images.iterdir() if p.is_file()]
    
    if not image_list:
        print(f"Warning: No images found in {images}. Skipping...")
        continue
        
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)
    
    print(f"Running LoFTR matching for '{folder_name}'...")
    features, matches = match_dense.main(
        conf=loftr_conf, 
        pairs=sfm_pairs, 
        image_dir=images, 
        export_dir=outputs
    )
    
    print(f"Starting COLMAP Reconstruction for '{folder_name}'...")
    reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=images,
        pairs=sfm_pairs,
        features=features,
        matches=matches
    )
    
    print(f"Finished processing '{folder_name}'!")

print("\nAll folders processed successfully!")