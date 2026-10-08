import cv2
import numpy as np
import random

def random_crop(image, boxes, labels, min_scale=0.3):
    h, w, c = image.shape
    scale_choices = np.arange(min_scale, 1.0, 0.1)
    box_center_x = (boxes[:, 0] + boxes[:, 2]) / 2
    box_center_y = (boxes[:, 1] + boxes[:, 3]) / 2
    for i, scale in enumerate(scale_choices):
        new_w = int(w * scale)
        new_h = int(h * scale)

        for _ in range(10):
            index = np.random.randint(0, len(boxes))
            center_x = box_center_x[index]
            center_y = box_center_y[index]

            crop_x1 = int(max(center_x - new_w / 2, 0))
            crop_y1 = int(max(center_y - new_h / 2, 0))
            crop_x2 = crop_x1 + new_w
            crop_y2 = crop_y1 + new_h
            if crop_x2 <= w and crop_y2 <= h:
                new_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
                new_boxes = boxes.copy()
                new_labels = labels.copy()
                new_boxes[:, [0, 2]] = boxes[:, [0, 2]] - crop_x1
                new_boxes[:, [1, 3]] = boxes[:, [1, 3]] - crop_y1
                new_boxes[:, [0, 2]] = np.clip(new_boxes[:, [0, 2]], 0, new_w)
                new_boxes[:, [1, 3]] = np.clip(new_boxes[:, [1, 3]], 0, new_h)
                new_boxes = new_boxes[np.where((new_boxes[:, 2] - new_boxes[:, 0]) > 0)]
                if len(new_boxes) == 0:
                    continue
                new_labels = new_labels[:len(new_boxes)]

                return new_image, new_boxes, new_labels

    return image, boxes, labels

def random_flip(image, boxes, labels):
    if np.random.random() < 0.5:
        image = cv2.flip(image, 1)
        boxes[:, [0, 2]] = image.shape[1] - boxes[:, [2, 0]]
    if np.random.random() < 0.5:
        image = cv2.flip(image, 0)
        boxes[:, [1, 3]] = image.shape[0] - boxes[:, [3, 1]]
    return image, boxes, labels

def random_rotation(image, boxes, labels, angle_range=(-45, 45)):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, scale=1)
    image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    n_boxes = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        coords = np.stack([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        coords = np.concatenate([coords, np.ones((4, 1))], axis=1)
        rotated_coords = np.dot(coords, rotation_matrix.T)
        x1 = np.min(rotated_coords[:, 0])
        y1 = np.min(rotated_coords[:, 1])
        x2 = np.max(rotated_coords[:, 0])
        y2 = np.max(rotated_coords[:, 1])
        if x2 - x1 > 10 and y2 - y1 > 10:
            n_boxes.append([x1, y1, x2, y2])
    if len(n_boxes) == 0:
        return image, np.zeros((0, 4)), np.zeros((0,))
    boxes = np.array(n_boxes)

    return image, boxes, labels


def random_color(image):
    alpha = random.uniform(0.5, 1.5)
    beta = random.uniform(-50, 50)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    hue_shift = random.uniform(-10, 10)
    saturation_shift = random.uniform(0.7, 1.3)
    value_shift = random.uniform(0.7, 1.3)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image_hsv[:, :, 0] = (image_hsv[:, :, 0] + hue_shift) % 180
    image_hsv[:, :, 1] = image_hsv[:, :, 1] * saturation_shift
    image_hsv[:, :, 2] = image_hsv[:, :, 2] * value_shift
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    return image