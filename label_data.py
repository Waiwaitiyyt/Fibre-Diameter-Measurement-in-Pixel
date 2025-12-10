import pickle
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

DATA_FILE = "data/raw_contours.pkl"
OUTPUT_FILE = "data/labeled_contours.pkl"

# Here is to obtain an image for a single contour for user to label it
# This function create a just-big-enough mask for the contour
def contourIsolate(contour: np.ndarray) -> np.ndarray:
    pts = contour.reshape(-1, 2)           
    x0, y0 = pts.min(axis=0).astype(int)
    x1, y1 = pts.max(axis=0).astype(int)
    h, w = (y1 - y0 + 50), (x1 - x0 + 50)
    mask = np.zeros((h, w), dtype=np.uint8)
    normalised = (25 + pts - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(mask, [normalised], -1, 255, 1)
    return mask

if __name__ == "__main__":

    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    labeled = []
    for i, item in enumerate(data):
        contour = item['contour']
        area = item['area']

        # Label contour shorter than 15 points to noise, as it's not possible to have a contour smaller than that
        if len(contour) <= 15:
            item['label'] = 0
            labeled.append(item)
            print(f"Contour {i} labelled as noise, length = {len(contour)}")
            continue  

        singleContourImg = contourIsolate(contour)
        cv2.imshow(f"Contour No.{i}" ,singleContourImg)

        # labelling
        # 0 - other stuff£¬ 1 - unwanted holes
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('0'):
            item['label'] = 0
            labeled.append(item)
            print(f"Contour {i} labelled as noise")
            i += 1
        elif key == ord('1'):
            item['label'] = 1
            labeled.append(item)
            print(f"Contour {i} labelled as pore")
            i += 1
        else:
            print(f"Invalid: {chr(key) if 32 <= key < 127 else key}. only 0/1/q.")
        cv2.destroyAllWindows()

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(labeled, f)
    print(f"Labelling finished! {len(labeled)} samples labelled, saved to{OUTPUT_FILE}")