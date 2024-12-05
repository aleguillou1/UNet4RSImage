# U-Net

This repository provides a simple and reliable tool to perform land cover classification using a U-Net model. A tutorial is provided here, but the notebook is flexible and can be adapted to your own datasets, satellite images, and labels. The only requirements are a Python environment capable of running Jupyter notebooks and QGIS.

The method requires as input a .tif image (the number of channels is adjustable), its corresponding label in .tif format, and a shapefile grid (Figure []).

![Description de l'image](./Fig/Label.png)

Step 1 : Patch extraction 

## Step 1: Patch Extraction

The **Patch Extraction** process is explained in detail in the notebook under the section **"Patch Extraction"** (look for the header with this name).

👉 **[Go to the Notebook](./Toolbox/Unet_tutorial.ipynb)** to find this section.

## Step 1: Patch Extraction

To prepare the training, validation, and test datasets for the U-Net model, you need to extract image patches and their corresponding labels. This process divides large geospatial images into smaller, more manageable sections.

---

### **Code Example**

Below is the Python function to perform patch extraction. It:
- Reads the satellite images, labels, and grid files.
- Divides the data into training, validation, and test sets.
- Extracts patches and saves them in respective directories.

```python
def extract_patches(image_path, label_path, shapefile_path, output_dir, train_ratio=0.75, val_ratio=0.2, test_ratio=0.05):
    import os
    import rasterio
    import geopandas as gpd
    import random
    import numpy as np
    import cv2
    from rasterio.mask import mask
    
    # Open the image and label files
    with rasterio.open(image_path) as src_img:
        with rasterio.open(label_path) as src_lbl:
            grid = gpd.read_file(shapefile_path)

            # Directory setup
            train_img_dir = os.path.join(output_dir, 'train/images')
            val_img_dir = os.path.join(output_dir, 'validation/images')
            test_img_dir = os.path.join(output_dir, 'test/images')
            train_lbl_dir = os.path.join(output_dir, 'train/labels')
            val_lbl_dir = os.path.join(output_dir, 'validation/labels')
            test_lbl_dir = os.path.join(output_dir, 'test/labels')
            
            for dir_path in [train_img_dir, val_img_dir, test_img_dir, train_lbl_dir, val_lbl_dir, test_lbl_dir]:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

            # Shuffle and split the patches
            indices = list(grid.index)
            random.shuffle(indices)
            num_patches = len(indices)
            num_train = int(train_ratio * num_patches)
            num_val = int(val_ratio * num_patches)
            train_indices = indices[:num_train]
            val_indices = indices[num_train:num_train + num_val]
            test_indices = indices[num_train + num_val:]
            
            # Helper to save patches
            def save_patch_and_label(patch, label, transform, output_img_subdir, output_lbl_subdir, patch_filename):
                patch_img_path = os.path.join(output_img_subdir, patch_filename)
                with rasterio.open(
                    patch_img_path, 'w',
                    driver='GTiff',
                    height=224, width=224,
                    count=4, dtype=patch.dtype, crs=src_img.crs, transform=transform
                ) as dst_img:
                    dst_img.write(patch)
                
                patch_lbl_path = os.path.join(output_lbl_subdir, patch_filename)
                with rasterio.open(
                    patch_lbl_path, 'w',
                    driver='GTiff',
                    height=224, width=224,
                    count=1, dtype=label.dtype, crs=src_lbl.crs, transform=transform
                ) as dst_lbl:
                    dst_lbl.write(label[0, :, :], 1)
            
            # Process each patch
            for idx, row in grid.iterrows():
                geom = [row['geometry']]
                patch_img, transform = mask(src_img, geom, crop=True)
                patch_lbl, _ = mask(src_lbl, geom, crop=True)

                patch_filename = f"patch_{idx}.tif"
                if idx in train_indices:
                    save_patch_and_label(patch_img, patch_lbl, transform, train_img_dir, train_lbl_dir, patch_filename)
                elif idx in val_indices:
                    save_patch_and_label(patch_img, patch_lbl, transform, val_img_dir, val_lbl_dir, patch_filename)
                elif idx in test_indices:
                    save_patch_and_label(patch_img, patch_lbl, transform, test_img_dir, test_lbl_dir, patch_filename)

print("Patch extraction complete!")

## Step 2: Image pre-processing

This cells aims to prepare the dataset for the training of the model. The cells will read your dataset, normalize them, transform them as tensor, you can also visualise a dataset to check if you patch and labels are the same for the training.
