import cv2
import numpy as np
import sys
from keras.models import load_model

crop_size = 44
def sliding_windows(image):
    height, width = image.shape

    step_x = 4
    step_y = 4

    # number of windows in x/y
    nx_windows = int((width - 44) / step_x)
    ny_windows = int((height - 44) / step_y)

    windows = []
    for i in range(ny_windows):
        for j in range(nx_windows):
            # calculate window position
            start_x = j * step_x
            end_x = start_x + crop_size
            start_y = i * step_y
            end_y = start_y + crop_size
            # append window to the list of windows
            windows.append(((start_x, start_y), (end_x, end_y)))
    return windows


def filter_windows(img, windows, model):
    # create empty list to receive positive detection windows
    windows_filtered = []
    # iterate over all windows
    for window in windows:
        t_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        t_img = t_img.reshape(1, 1, 44, 44).astype('float32')
        t_img = t_img / 255.0
        pred_probability = model.predict(t_img, batch_size=1)

        if pred_probability[0][0] > 0.7:
            windows_filtered.append(window)
    return windows_filtered


def predict_phone_position(image_dir, model):
    image = cv2.imread(image_dir)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    windows = sliding_windows(image)

    windows_filtered = filter_windows(image, windows, model)

    if len(windows_filtered) == 0:
        return [0, 0]
    heatmap = np.zeros_like(image[:, :]).astype(np.float)

    for w in windows_filtered:
        heatmap[w[0][1]: w[1][1], w[0][0]:w[1][0]] += 1
    heatmap = heatmap.astype('uint8')

    im2, contours, hierarchy = cv2.findContours(heatmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # find the biggiest area
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area > 10000:
        desc_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cont in desc_contours:
            if cv2.contourArea(cont) < 10000:
                c = cont
                break
    x, y, w, h = cv2.boundingRect(c)

    bbox = [(x, y), (x + w, y + h)]

    return [round((float(bbox[0][0] + bbox[1][0]) / 2) / image.shape[1], 4),
            round((float(bbox[0][1] + bbox[1][1]) / 2) / image.shape[0], 4)]

def main():
    image_dir = sys.argv[1]
    # Load CNN model from disk
    model = load_model('model.h5')
    # print "Model loaded from disk!"
    pos = predict_phone_position(image_dir, model)
    print(pos[0], " ", pos[1])

if __name__ == "__main__":
    main()
