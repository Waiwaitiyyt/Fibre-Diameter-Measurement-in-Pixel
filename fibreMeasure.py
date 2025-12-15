import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from typing import Tuple
from scipy.ndimage import distance_transform_edt
from predict import predict_single_contour
import torch
from model import PointNetLike
from typing import Tuple

def _distanceMeasure(binary_mask: np.ndarray,
                     min_area=50,
                     width_jump_ratio=1.5,
                     local_window=10,
                     max_radius_factor=1.8) -> Tuple[list, np.ndarray, list]:
    """
    width_jump_ratio : reject if local width deviates too much from neighbors
    local_window     : neighborhood size along skeleton
    max_radius_factor: reject points that are too thick relative to median
    """

    fg = (binary_mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    large_mask = np.zeros_like(fg)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            large_mask[labels == i] = 1
    if large_mask.sum() == 0:
        return [], None, []

    dt = distance_transform_edt(large_mask)
    skel_bool = skeletonize(large_mask > 0)
    skelImg = (skel_bool.astype(np.uint8) * 255)

    # --- Neighbor count (junction & endpoint removal) ---
    padded = np.pad(skel_bool.astype(np.uint8), 1)
    neighbor_counts = np.zeros_like(skel_bool, dtype=np.int32)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for dy, dx in offsets:
        neighbor_counts += padded[1+dy:1+dy+skel_bool.shape[0],
                                  1+dx:1+dx+skel_bool.shape[1]]

    # Keep only simple chain points (no junctions, no endpoints)
    keep_mask = skel_bool & (neighbor_counts == 2)

    coords = np.argwhere(keep_mask)
    if coords.size == 0:
        return [], skelImg

    widths = 2.0 * dt[keep_mask]

    # --- Global thickness sanity check ---
    median_width = np.median(widths)
    valid = widths < (max_radius_factor * median_width)

    coords = coords[valid]
    widths = widths[valid]

    if len(widths) < local_window:
        return [], skelImg

    # --- Local width consistency filter ---
    # Sort skeleton points by approximate arc length (using y,x order is OK locally)
    order = np.lexsort((coords[:,1], coords[:,0]))
    widths = widths[order]

    good = np.ones(len(widths), dtype=bool)
    half = local_window // 2

    for i in range(half, len(widths) - half):
        local = widths[i-half:i+half+1]
        local_med = np.median(local)
        if widths[i] > width_jump_ratio * local_med:
            good[i] = False
        if widths[i] < local_med / width_jump_ratio:
            good[i] = False

    widths = widths[good]
    if len(widths) > 2000:
        idx = np.random.choice(len(widths), 2000, replace=False)
        widths = widths[idx]
        coords = coords[idx]

    return widths.tolist(), skelImg, coords.tolist()


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

    imgCopy = cleanImg.copy()
    border_color = 255   
    border_thickness = 1 
    h, w = imgCopy.shape[:2]
    cv2.rectangle(imgCopy, (0, 0), (w - border_thickness - 1, h - 1 - border_thickness), border_color, border_thickness)
    # cv2.line(cleanImg, (0, h - 1), (w - 1, h - 1), border_color, 2 * border_thickness)
    contours, _ = cv2.findContours(imgCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, contour in enumerate(contours):
        contour = contour.squeeze(1)
        cls, conf = predict_single_contour(model, contour, DEVICE, NUM_POINTS)
        if cls == 1:
            cv2.drawContours(cleanImg, [contour], -1, 255, -1)

    return cleanImg

def setup():
    global DEVICE, NUM_POINTS, model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "pointnet_pore_classifier.pth" 
    NUM_POINTS = 64
    print(f"Device: {DEVICE}")
    print("Loading model...")
    model = PointNetLike(num_classes=2, num_points=NUM_POINTS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

def measure(imgPath):
    cleanImg = _imgProcess(imgPath)
    diameterList, skeleton, sampleCoordsList = _distanceMeasure(cleanImg)
    averageDiameter = np.mean(diameterList)
    measuredImg = cv2.cvtColor(cleanImg, cv2.COLOR_GRAY2BGR)
    skeletonIndice = np.where(skeleton == 255)
    measuredImg[skeletonIndice] = (0, 0, 255)
    for coord in sampleCoordsList:
        cv2.circle(measuredImg, (coord[1], coord[0]), 3, (255, 0, 0), -1)
    return averageDiameter, measuredImg


if __name__ == "__main__":
    imgPath = r"images\\2.JPG"
    setup()
    averageDiameter, measuredImg = measure(imgPath)
    cv2.imshow("a", measuredImg)  
    cv2.imwrite("demo.jpg", measuredImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(averageDiameter)
