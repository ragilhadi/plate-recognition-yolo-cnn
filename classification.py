import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

MAX_DIAG_MULTIPLYER = 5 
MAX_ANGLE_DIFF = 12.0 
MAX_AREA_DIFF = 0.5 
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3 # 3
possible_contours = []

def find_chars(contour_list):
    matched_result_idx = []
    
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER             and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF             and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # append this contour
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        
        # recursive
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx


def process_segmentation_char(img, model):
    label = ''
    classes = ["0","1","2","3","4","5","6","7","8","9","F"]
    height, width, channel = img.shape
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blurred = cv.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    img_thresh = cv.adaptiveThreshold(
        img_blurred, 
        maxValue=255.0, 
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv.THRESH_BINARY_INV, 
        blockSize=19, 
        C=9
    )
    contours, _= cv.findContours(
        img_thresh, 
        mode=cv.RETR_LIST, 
        method=cv.CHAIN_APPROX_SIMPLE
    )
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    cv.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
        # insert to dict
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })
    
    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0


    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
    
        if area > MIN_AREA \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
        
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
    
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        cv.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
    
    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            x = d['x']
            y = d['y']
            w = d['x']+d['w']
            z = d['y']+d['h']
            dim = (30,30)
            cropped_charachter = img[y:z, x:w]
            cropped_charachter = cv.resize(cropped_charachter, dim, interpolation = cv.INTER_AREA)
            input_img = np.asarray(cropped_charachter).reshape((1, 30, 30, 3))
            label = label + str(classes[np.argmax(model.predict(input_img))])
            
    label_cut = label[:3]


    return label_cut