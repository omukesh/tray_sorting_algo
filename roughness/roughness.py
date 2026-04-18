import cv2

import numpy as np

from skimage.feature import graycomatrix, graycoprops



def get_ultra_robust_score(patch):

    """

    Analyzes texture while ignoring structured edges.

    Includes a fix for the ValueError level mismatch.

    """

    # 1. Morphological Opening: Removes thin lines/text

    kernel = np.ones((3,3), np.uint8)

    patch_clean = cv2.morphologyEx(patch, cv2.MORPH_OPEN, kernel)



    # 2. Difference of Gaussians: Isolates texture frequency

    d = cv2.absdiff(cv2.GaussianBlur(patch_clean, (3,3), 0), 

                    cv2.GaussianBlur(patch_clean, (15,15), 0))



    # 3. GLCM Quantization Fix

    num_levels = 16

    bins = np.linspace(0, 256, num_levels + 1)

    # np.digitize returns 1-based indices, we subtract 1 to get 0-15

    q = np.digitize(d, bins) - 1

    # CRITICAL FIX: Ensure no value exceeds 15

    q = np.clip(q, 0, num_levels - 1)

    

    glcm = graycomatrix(q, [1], [0, np.pi/4, np.pi/2], 

                        levels=num_levels, symmetric=True, normed=True)

    

    contrast = graycoprops(glcm, 'contrast').mean()

    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    

    return (contrast * (1.0 - homogeneity)) + 1e-7



def process_metal_surface(image_path):

    src = cv2.imread(image_path)

    if src is None:

        print(f"Error: Image not found at {image_path}")

        return



    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))

    gray = clahe.apply(gray)



    h, w = gray.shape

    win_size, step = 40, 10

    

    map_h, map_w = (h - win_size) // step + 1, (w - win_size) // step + 1

    rough_map = np.zeros((map_h, map_w), dtype=np.float32)



    print(f"Scanning {map_h * map_w} surface patches...")

    

    for i in range(map_h):

        for j in range(map_w):

            y, x = i * step, j * step

            patch = gray[y : y + win_size, x : x + win_size]

            rough_map[i, j] = get_ultra_robust_score(patch)



    # REFINEMENT & BLENDING

    rough_map_smooth = cv2.medianBlur(rough_map, 5)

    norm_map = cv2.normalize(np.sqrt(rough_map_smooth), None, 0, 1, cv2.NORM_MINMAX)

    heatmap = cv2.resize(norm_map, (w, h), interpolation=cv2.INTER_LANCZOS4)



    pink_tint = np.array([180, 50, 255], dtype=np.float32) 

    alpha = np.expand_dims(np.power(heatmap, 1.2), axis=2).astype(np.float32)

    

    img_float = src.astype(np.float32)

    blended = (img_float * (1.0 - alpha)) + (pink_tint * alpha)

    result = np.clip(blended, 0, 255).astype(np.uint8)



    cv2.imshow('Final Corrected Roughness Map', result)

    cv2.waitKey(0)

    cv2.destroyAllWindows()



if __name__ == "__main__":

    process_metal_surface('/home/mdl/Downloads/r1.png')