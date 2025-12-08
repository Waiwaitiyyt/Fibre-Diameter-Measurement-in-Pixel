import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, reconstruction, remove_small_holes, remove_small_objects
from skimage.draw import line
from typing import Tuple
from scipy.ndimage import distance_transform_edt



def _distanceMeasure(binary_mask: np.ndarray, min_area=50) -> list:
    fg = (binary_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    large_mask = np.zeros_like(fg)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            large_mask[labels == i] = 1
    dt = distance_transform_edt(large_mask)
    skel_bool = skeletonize(large_mask > 0)
    skel_indices = np.argwhere(skel_bool)

    # # Skeletonize visualisation
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(cleanImg, cmap='gray')
    # plt.title('Original Binary Image')
    # plt.subplot(1, 2, 2)
    # plt.imshow(skel_bool, cmap='gray')
    # plt.title('Skeletonized Image')
    # plt.show()

    # Exclude endpoints and junctions: compute neighbor counts
    # pad skeleton to avoid boundary checks
    padded = np.pad(skel_bool.astype(np.uint8), pad_width=1, mode='constant', constant_values=0)
    neighbor_counts = np.zeros_like(skel_bool, dtype=np.int32)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for dy, dx in offsets:
        neighbor_counts += padded[1+dy:1+dy+skel_bool.shape[0], 1+dx:1+dx+skel_bool.shape[1]]

    keep_mask = (skel_bool & (neighbor_counts == 2))

    coords = np.argwhere(keep_mask)
    if coords.size == 0:
        return []
    widths = 2.0 * dt[keep_mask]
    # optionally subsample to reduce correlation (random or regular)
    # e.g., take every nth sample or random subset
    if len(widths) > 2000:
        idx = np.random.choice(len(widths), size=2000, replace=False)
        widths = widths[idx]
    return widths.tolist()

def _contourProperty(contour: np.ndarray) -> Tuple[int, int]:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect = max(w, h) / (min(w, h) + 1e-6)
    return area, aspect

def _imgProcess(imgPath: str) -> np.ndarray:
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
    areaThreshold, aspectThreshold = 1e4, 2.1 # Obtained via some measurement of mine
    cleanBinaryImg = cleanImg.copy()
    border_color = 255   
    border_thickness = 1         
    h, w = cleanImg.shape[:2]
    cv2.line(cleanImg, (0, 0), (w, 0), border_color, border_thickness)
    cv2.line(cleanImg, (0, h-1), (w, h-1), border_color, border_thickness)
    cv2.line(cleanImg, (0, 0), (0, h), border_color, border_thickness)
    cv2.line(cleanImg, (w-1, 0), (w-1, h), border_color, border_thickness)
    contours, _ = cv2.findContours(cleanImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, contour in enumerate(contours):
        area,aspect = _contourProperty(contour)
        if area < areaThreshold and aspect > aspectThreshold:
            cv2.drawContours(cleanBinaryImg, [contour], -1, 255, -1)

    return cleanBinaryImg 

def measure(imgPath):
    cleanBinaryImg = _imgProcess(imgPath)
    cv2.imshow("a", cleanBinaryImg)
    cv2.waitKey(0)
    cv2.destroyWindow("a")
    averageDiameter = np.mean(_distanceMeasure(cleanBinaryImg))
    return averageDiameter


if __name__ == "__main__":
    imgPath = r"images\\8.JPG"
    cleanImg = _imgProcess(imgPath)  
    cv2.imshow(imgPath, cleanImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    diameters_px = _distanceMeasure(cleanImg, min_area=500) 
    if diameters_px:
        print("mean (px):", np.mean(diameters_px))
