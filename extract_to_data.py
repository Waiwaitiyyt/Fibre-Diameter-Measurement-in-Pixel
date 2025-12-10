import os
import cv2
import numpy as np
import pickle
from skimage.morphology import remove_small_holes, label

IMAGE_DIR = "images"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def imgProcess(imgPath: str) -> np.ndarray:
    grayImg = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    binary = cv2.threshold(cv2.bitwise_not(grayImg), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernelSize = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # Remove small objects
    mask = np.zeros_like(closing)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closing, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 5 * kernelSize ** 2:
            mask[labels == i] = 255
    # Remove small holse
    invImg = cv2.bitwise_not(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(invImg, connectivity=8)
    filterImg = np.zeros_like(invImg)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 5 * kernelSize ** 2:
            filterImg[labels == i] = 255

    # Some gaps and pores are filled in the previous step, the result would be theoretically greater than the actual value
    # Thus some small holes remained could be the compensation for the bias above.
    # Well actually I just dunno how to perfectly extract the overlapping fibre contour
    cleanImg = cv2.bitwise_not(filterImg)
    # Here is a re-process for the binary image
    # Contours are extarcted to distinguish the unwanted hole and remove them
    cleanBinaryImg = cleanImg.copy()
    border_color = 255   
    border_thickness = 1         
    h, w = cleanImg.shape[:2]
    cv2.line(cleanImg, (0, 0), (w, 0), border_color, border_thickness)
    cv2.line(cleanImg, (0, h-1), (w, h-1), border_color, border_thickness)
    cv2.line(cleanImg, (0, 0), (0, h), border_color, border_thickness)
    cv2.line(cleanImg, (w-1, 0), (w-1, h), border_color, border_thickness)
    # contours, _ = cv2.findContours(cleanImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


    return cleanBinaryImg 


if __name__ == "__main__":
    all_items = []
    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = imgProcess(img_path)
        if img is None:
            continue

        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            equiv_diam = np.sqrt(4 * area / np.pi)
            item = {
                'contour': cnt.reshape(-1, 2).astype(np.float32),
                'area': float(area),
                'perimeter': float(perimeter),
                'equiv_diam': float(equiv_diam),
                'label': -1  # Label to be -1 as default, will be re-labelled in label_data.py c
            }
            all_items.append(item)
    # save as .pkl
    output_file = os.path.join(OUTPUT_DIR, "raw_contours.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(all_items, f)
    print(f"Extracted {len(all_items)} contours, saved to {output_file}")