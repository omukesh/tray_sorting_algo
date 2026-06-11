import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def roughness(patch):
    blur1 = cv2.GaussianBlur(patch, (3, 3), 0)
    blur2 = cv2.GaussianBlur(patch, (9, 9), 0)
    texture_only = cv2.absdiff(blur1, blur2)

    # B. Statistical Texture (GLCM)
    # We use 'Correlation' and 'Contrast' - correlation helps ignore uniform glare
    bins = np.linspace(0, 256, 16)
    quantized = np.digitize(texture_only, bins) - 1
    glcm = graycomatrix(quantized, [1], [0, np.pi/4], levels=16, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    # C. Edge Frequency
    # Sobel filters are more robust to direction than Laplacian
    sobelx = cv2.Sobel(texture_only, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(texture_only, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    edge_density = np.mean(mag)

    # combined Score: Contrast is weighted by (1 - correlation)
    # If pixels are highly correlated, it's likely a glare, not roughness.
    score = edge_density * (contrast / (correlation + 0.1))
    return score

# --- Setup and Pre-processing ---
image = cv2.imread('/home/mdl/Downloads/r1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# lighting normalization
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16,16))
gray = clahe.apply(gray)

h, w = gray.shape
win_size = 48  # increased window size for better statistical stability
step = 12      # smaller step for smoother heatmap
rough_map = np.zeros((h // step, w // step), dtype=np.float32)

print("running roughness algo...")
for i, y in enumerate(range(0, h - win_size, step)):
    for j, x in enumerate(range(0, w - win_size, step)):
        patch = gray[y:y+win_size, x:x+win_size]
        rough_map[i, j] = roughness(patch)

# --- Heatmap Generation ---
# Log scale often works better for roughness because it grows exponentially
rough_map = np.log1p(rough_map) 
norm_map = cv2.normalize(rough_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
heatmap_gray = cv2.resize(norm_map, (w, h), interpolation=cv2.INTER_LANCZOS4)

# Build the Output
output = image.copy()
pink_tint = np.array([180, 50, 255], dtype=np.uint8) # BGR Pink

# Create a smoother blend
alpha = (heatmap_gray / 255.0)[:, :, np.newaxis]
alpha = np.power(alpha, 1.5) 

result = (image * (1 - alpha) + pink_tint * alpha).astype(np.uint8)

cv2.imshow('Roughness', result)
cv2.waitKey(0)
