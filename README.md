# U-Net

This repository provides a simple and reliable tool to perform land cover classification using a U-Net model. A tutorial is provided here, but the notebook is flexible and can be adapted to your own datasets, satellite images, and labels. The only requirements are a Python environment capable of running Jupyter notebooks and QGIS.

The method requires as input a .tif image (the number of channels is adjustable), its corresponding label in .tif format, and a shapefile grid (Figure []).

IMPORTANT : In our cas we use a Pléiades images (50cm resolution pixel) et for the label we use OCS GE (freely avaible here: https://geoservices.ign.fr/ocsge). This repo aims to propose a modulable approach for differente data (different resolution and labels/Classes). You can use RGB Image or Multi-spectral imagery, you can also choose how much classes you want to predict. For the moment we propose only one U-Net architecture who ask a 224x224xn_channels and labels in 224x224x1 (in bytes, Int8) format as input. 

Before start, you need to have a satellite image in tif format, a label image in the same position than your satellite image (same projection), you also need to have a grid to extract your patchs. You can directly create your own grid from the Qgis tools : "Create Grids", you need to have a grids who fit (at least closesly) to make 224x224 pixels patchs.
For examples, in our case we use Pléaides images (From Airbus DS) on 50cm resolution format so to make a 224x224 grids patchs we have to founs how much meters we need to put to have a 224 pixels in longueur and largeur so :  224/2 = 112 , so we need to create a grid with 112 meter in longueur and largeur to have a 224x224 patches. 


| ![Image 1](./Fig/Label.png) | ![Image 2](./Fig/Image.png) |
|:--------------------------------:|:--------------------------------:|
| **Figure 1**: Pléiades image     | **Figure 2**: Label from OCS GE     |



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

```

### Step 2: Image pre-processing

This cells aims to prepare the dataset for the training of the model. The cells will read your dataset, normalize them, transform them as tensor, you can also visualise a dataset to check if you patch and labels are the same for the training.
```python

# Normalization
def normalize_img(img):
    return img / np.max(img)

# Function to read .tif files and convert them into tensors
def load_data(image_dir, mask_dir):
    images = []
    masks = []
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.tif') or img_file.endswith('.tiff'):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)
            
            # Read images and masks
            img = tiff.imread(img_path).astype(np.float32)
            mask = tiff.imread(mask_path).astype(np.uint8)
            
            img = normalize_img(img)
            
            images.append(img)
            masks.append(mask)
    
    # Convert the lists into numpy arrays
    images = np.array(images)
    masks = np.array(masks)
    
    # Convert numpy arrays into tensors
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    masks = tf.convert_to_tensor(masks, dtype=tf.uint8)
    
    return images, masks

train_image_dir = '/Patch/train/images'
train_mask_dir = '/Patch/train/labels'
val_image_dir = '/Patch/validation/images/'
val_mask_dir = '/Patch/validation/labels'


# Load and transform the training data
train_images, train_masks = load_data(train_image_dir, train_mask_dir)
val_images, val_masks = load_data(val_image_dir, val_mask_dir)

print(f'Shape of training images: {train_images.shape}')
print(f'Shape of training labels: {train_masks.shape}')

# Check the pixel values of the images
print(f'Minimum and maximum values of the training images: {tf.reduce_min(train_images).numpy()}, {tf.reduce_max(train_images).numpy()}')
print(f'Minimum and maximum values of the training labels: {tf.reduce_min(train_masks).numpy()}, {tf.reduce_max(train_masks).numpy()}')

print(f'Minimum and maximum values of the validation images: {tf.reduce_min(val_images).numpy()}, {tf.reduce_max(val_images).numpy()}')
print(f'Minimum and maximum values of the validation labels: {tf.reduce_min(val_masks).numpy()}, {tf.reduce_max(val_masks).numpy()}')

# Display an RGB image and its corresponding mask
idx = 17  # You can change this index to visualize other images
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(train_images[idx][:, :, :3])  
plt.title('Image d\'entraînement (RGB)')

plt.subplot(1, 2, 2)
plt.imshow(train_masks[idx], cmap='gray') 
plt.title('Masque d\'entraînement')

plt.show()

```

