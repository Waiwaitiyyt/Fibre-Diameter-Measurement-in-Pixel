# predict.py
import torch
import numpy as np
import cv2
from model import PointNetLike
from scipy.interpolate import interp1d

# --- Helper Functions ---
def resample_contour(contour, num_points=64):
    n_points = len(contour)
    if n_points == 0:
        raise ValueError("Contour has zero points.")
    if n_points < num_points:
        padding_needed = num_points - n_points
        repeated_last_point = np.tile(contour[-1], (padding_needed, 1))
        contour = np.vstack([contour, repeated_last_point])
    elif n_points > num_points:
        old_indices = np.arange(n_points)
        new_indices = np.linspace(0, n_points - 1, num_points)
        fx = interp1d(old_indices, contour[:, 0], kind='linear')
        fy = interp1d(old_indices, contour[:, 1], kind='linear')
        contour = np.column_stack((fx(new_indices), fy(new_indices)))
    return contour

def normalize_contour(contour):
    center = contour.mean(axis=0)
    contour_centered = contour - center
    max_dist = np.max(np.linalg.norm(contour_centered, axis=1))
    if max_dist == 0: max_dist = 1e-8
    return contour_centered / max_dist

def compute_scalars(contour):
    cnt_int = contour.astype(np.int32).reshape(-1, 1, 2)
    area = cv2.contourArea(cnt_int)
    perimeter = cv2.arcLength(cnt_int, closed=True)
    min_coords = contour.min(axis=0)
    max_coords = contour.max(axis=0)
    diameter = np.linalg.norm(max_coords - min_coords)
    return np.array([area, perimeter, diameter], dtype=np.float32)


def predict_single_contour(model, contour, device, num_points=64):
    # Re-sample
    contour_rs = resample_contour(contour, num_points=num_points)
    
    # Normalisation
    contour_norm = normalize_contour(contour_rs)
    
    # Obtain scalars
    scalars_np = compute_scalars(contour_rs)
    
    # Tensor transformation
    points_tensor = torch.tensor(contour_norm, dtype=torch.float32).unsqueeze(0).to(device) # (1, N, 2)
    scalars_tensor = torch.tensor(scalars_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 3)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(points_tensor, scalars_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    return pred_class, confidence

# --- Main Prediction Logic ---
def extract_contours_from_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    contours, _ = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contour_points_list = [cnt.squeeze(1) for cnt in contours if len(cnt) > 5] 
    return contour_points_list

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
    cleanImg = cv2.bitwise_not(filterImg)
    border_color = 255   
    border_thickness = 1
    h, w = cleanImg.shape[:2]
    cv2.rectangle(cleanImg, (0, 0), (w - 1, h - 1), border_color, border_thickness)
    return cleanImg 


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "pointnet_pore_classifier.pth" # Use the name from train.py
    NUM_POINTS = 64
    print(DEVICE)
    print("Loading model...")
    model = PointNetLike(num_classes=2, num_points=NUM_POINTS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
    IMAGE_PATH = "images\\A.JPG" 
    try:
        cleanBinaryImg = _imgProcess(IMAGE_PATH)
        mask = cleanBinaryImg.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for i, contour in enumerate(contours):
            contour = contour.squeeze(1)
            cls, conf = predict_single_contour(model, contour, DEVICE, NUM_POINTS)
            if cls == 1:
                cv2.drawContours(cleanBinaryImg, [contour], -1, 255, -1)

        cv2.imshow("a", cleanBinaryImg)
        cv2.imshow("b", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    except FileNotFoundError as e:
        print(e)
