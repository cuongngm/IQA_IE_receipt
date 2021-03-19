import cv2
import numpy as np
from PIL import Image
from scipy.spatial import distance as dist
from torchvision.transforms.functional import to_tensor
from scipy.signal import convolve2d
from scanner import four_point_transform, DocScanner


def localNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = (patch - patch_mean) / patch_std
    return patch_ln


def patchSifting(im, patch_size=24, stride=24):
    img = np.array(im).copy()
    im1 = localNormalization(img)
    im1 = Image.fromarray(im1)
    ret1, im2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    w, h = im1.size
    patches = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = im2[i:i + patch_size, j:j + patch_size]
            if judgeAllOnesOrAllZreos(patch):
                patch = to_tensor(im1.crop((j, i, j + patch_size, i + patch_size)))
                patch = patch.float().unsqueeze(0)
                patches = patches + (patch,)
                # patch = np.zeros(patch.shape, dtype=np.uint8)
                # im2[i: i + patch_size, j:j + patch_size] = patch
            # else:
                # patch = np.zeros(patch.shape, dtype=np.uint8)
                # patch.fill(255)
                # im2[i: i + patch_size, j:j + patch_size] = patch
    return patches


def judgeAllOnesOrAllZreos(patch):
    flag1 = True
    flag2 = True
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            if patch[i, j] == 255:
                continue
            else:
                flag1 = False

    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            if patch[i, j] == 0:
                continue
            else:
                flag2 = False
    return flag1 or flag2


def image_normalize(img):
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1.0/np.sqrt(img.size))
    result = np.multiply(np.subtract(img, mean), 1/std_adj)
    if len(result.shape) == 2:
        result = np.expand_dims(result, -1)
    return result


def is_rectangle(vertices):
    if len(vertices) != 4:
        return False
    for i in range(-1, 3):
        ver1 = vertices[i-1] - vertices[i]
        ver2 = vertices[i+1] - vertices[i]
        if dist.cosine(ver1, ver2) < 0.8:
            return False
    return True


def get_document_corners(img, img_type='BGR'):
    border_img = 40
    if img_type == 'BGR':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_type == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    origin_h, origin_w = img.shape
    # if origin_h < origin_w:
    #     img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    new_w = 600
    scale = origin_w/new_w
    new_h = int(origin_h/scale)
    img = cv2.resize(img, (new_w, new_h))
    corners_img_std = np.std(img)
    img = cv2.copyMakeBorder(img, border_img, border_img, border_img, border_img, cv2.BORDER_CONSTANT)
    img = cv2.medianBlur(img, 11)
    edges = cv2.Canny(img, 10, 30)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    cnts, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    simplified_cnts = []
    for cnt in cnts:
        hull = cv2.convexHull(cnt)
        simplified_cnts.append(cv2.approxPolyDP(hull, 0.05*cv2.arcLength(hull, True), True))
    simplified_cnts = sorted(simplified_cnts, key=cv2.contourArea, reverse=True)
    area_th = cv2.contourArea(simplified_cnts[0])/5
    corners = None
    for i in range(1, min(5, len(simplified_cnts))):
        cnt = simplified_cnts[i].reshape(-1, 2)
        if cv2.contourArea(cnt) < area_th:
            break
        if is_rectangle(cnt):
            current_img = four_point_transform(img, cnt)
            current_img_std = np.std(current_img)
            if corners is None or current_img_std*1.2 < corners_img_std:
                corners = cnt
                corners_img_std = current_img_std
    if corners is None:
        corners = simplified_cnts[0].reshape(-1, 2)
    corners -= border_img
    corners = (corners.reshape(-1, 2)*scale).astype(np.int32)
    return corners


def is_blank(patch, blank_threshold=1.0):
    # check patch is blank (no characters)
    compare_matrix = patch == 255
    if blank_threshold >= 1:
        return np.all(compare_matrix)
    else:
        blank_ratio = np.count_nonzero(compare_matrix)/compare_matrix.size
        return blank_ratio >= blank_threshold or blank_ratio <= 1 - blank_threshold


def generate_patches(img, patch_size=48, img_type='RGB', blank_threshold=0.1,
                     row_step=None, col_step=None, document_crop=True):
    img = np.array(img)
    img = img[:, :, ::-1]  # Pillow to cv2

    if document_crop:
        # first solutions
        # corners = get_document_corners(img, img_type)
        # img = four_point_transform(img, corners)
        scanner = DocScanner()
        img = scanner.scan(img)

    if img_type == 'BGR':
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img_type == 'RGB':
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img_type == 'GRAY':
        gray_img = img
    img = cv2.medianBlur(gray_img, 15)
    """
    thres = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 15, 2)
    patches = []
    rows, cols = gray_img.shape
    if row_step is None:
        row_step = int(patch_size[1]/3)
    if col_step is None:
        col_step = int(patch_size[0]/3)
    for i in range(0, rows-patch_size[1], row_step):
        for j in range(0, cols-patch_size[0], col_step):
            if not is_blank(thres[i:i+patch_size[1], j:j+patch_size[0]], blank_threshold=blank_threshold):
                patches.append(gray_img[i:i+patch_size[1], j:j+patch_size[0]])
    """
    im1 = Image.fromarray(img)
    _, im2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = img.shape[:2]  # (480, 720)
    patches = ()
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            if not is_blank(im2[i:i+patch_size, j:j+patch_size], blank_threshold=blank_threshold):
                patch = to_tensor(im1.crop((j, i, j+patch_size, i+patch_size)))
                patch = patch.float().unsqueeze(0)
                patches = patches + (patch, )
    return patches


if __name__ == '__main__':
    img = cv2.imread('a.jpg')
    patches = generate_patches(img)
    print(patches)
