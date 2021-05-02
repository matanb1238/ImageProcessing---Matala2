import time
from collections import defaultdict


import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt, pi, cos, sin



def conv1D(inSignal:np.ndarray,kernel1:np.ndarray)->np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """
    # Flip the kernel
    kernel1 = kernel1[::-1]
    # Match the arrays size (easier to conv like this)
    arraySignal = np.zeros(inSignal.size + kernel1.size - 1)
    arrayKernel = np.zeros(inSignal.size + kernel1.size - 1)
    finalArray = np.zeros(inSignal.size + kernel1.size - 1)
    for i in range(inSignal.size):
        arraySignal[i] = inSignal[i]
    for i in range(kernel1.size):
        arrayKernel[i + inSignal.size - 1] = kernel1[i]
    for i in range(arraySignal.size):
        sum = 0
        for j in range(arraySignal.size):
            sum += arraySignal[j]*arrayKernel[j]
        for k in range(arraySignal.size - 2, -1, -1):
            arraySignal[k+1] = arraySignal[k]
        arraySignal[0] = 0
        finalArray[i] = sum
    # My conv process is reversed (I iterated the signal over the kernel instead of the opposite) so I am flipping the answer
    return finalArray[::-1]

def conv2D(inImage:np.ndarray,kernel2:np.ndarray)->np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    1
    :return: The convolved image
    """
    # Flip the kernel
    kernel2 = np.flipud(np.fliplr(kernel2))
    # Convolution output
    output = np.zeros_like(inImage)

    # Row and col size
    k_row = len(kernel2)
    k_col = len(kernel2[0])
    add_r = int((k_row - 1) / 2)
    add_c = int((k_col - 1) / 2)

    # Padding
    padded_img = np.zeros((inImage.shape[0] + k_row - 1, inImage.shape[1] + k_col - 1))
    padded_img[k_row - 2:-(k_row - 2), k_col - 2:-(k_col - 2)] = inImage
    up_r = padded_img[add_r]
    down_r = padded_img[-add_r - 1]
    temp_padded = np.flipud(np.fliplr(padded_img))
    left_c = temp_padded[add_c]
    right_c = temp_padded[-add_c - 1]
    for i in range(add_r):
        padded_img[i] = up_r
        padded_img[-i] = down_r
    for j in range(add_c):
        for row in range(padded_img.shape[0]):
            padded_img[row, j] = left_c[row]
            padded_img[row, -j] = right_c[-row]

    # Multiply
    for x in range(inImage.shape[0]):
        for y in range(inImage.shape[1]):
            output[x, y] = (padded_img[x: x + k_col, y: y + k_row] * kernel2).sum()
            if kernel2.sum() != 0:
              output[x, y] /= kernel2.sum()

    return output

def convDerivative(inImage:np.ndarray) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray):
    x_kernel = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    y_kernel = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
    x_der = cv2.filter2D(inImage, -1, x_kernel, borderType=cv2.BORDER_REPLICATE)
    y_der = cv2.filter2D(inImage, -1, y_kernel, borderType=cv2.BORDER_REPLICATE)

    magnitude = pow(((pow(x_der, 2)) + pow(y_der, 2)), 0.5)
    EPS = 0.00000000001
    directions = np.arctan(np.divide(y_der, x_der+EPS))

    return directions, magnitude, x_der, y_der
"""
Calculate gradient of an image
:param inImage: Grayscale iamge
:return: (directions, magnitude,x_der,y_der)
"""

def blurImage1(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
    """
Blur an image using a Gaussian kernel
:param inImage: Input image
:param kernelSize: Kernel size
:return: The Blurred image
"""
    size = kernel_size[0]
    sigma = 1
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    in_image = cv2.filter2D(in_image, -1, g)
    return in_image

def blurImage2(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
    """
Blur an image using a Gaussian kernel using OpenCV built-in functions
:param inImage: Input image
:param kernelSize: Kernel size
:return: The Blurred image
"""
    return cv2.GaussianBlur(in_image, (kernel_size[0], kernel_size[1]), 0)

def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7)-> (np.ndarray, np.ndarray):
    # Gaussian blur
    img1 = cv2.GaussianBlur(img, (3, 3), 0)
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # Thresh normalization
    thresh = thresh * 255

    # Convolve the image with Gx, Gy
    smooth_x = cv2.filter2D(img1, -1, Gx, borderType=cv2.BORDER_REPLICATE).astype(float)
    smooth_y = cv2.filter2D(img1, -1, Gy, borderType=cv2.BORDER_REPLICATE).astype(float)
    my_sobel = np.sqrt(smooth_x ** 2 + smooth_y ** 2)

    cv_sobel = cv2.GaussianBlur(img, (3, 3), 0)
    def thresh1(img : np.ndarray, thresh : float = 0.7):
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x, y] >= thresh:
                    img[x, y] = 255
                else:
                    img[x, y] = 0
    sobel_x = cv2.Sobel(cv_sobel, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(cv_sobel, cv2.CV_64F, 0, 1, ksize=3)
    cv_ans = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Using the thresh
    thresh1(my_sobel, thresh)
    thresh1(cv_ans, thresh)

    return my_sobel, cv_ans
    """
Detects edges using the Sobel method
:param img: Input image
:param thresh: The minimum threshold for the edge response
:return: opencv solution, my implementation
"""
def edgeDetectionZeroCrossingSimple(img:np.ndarray)->(np.ndarray):
    """
Detecting edges using the "ZeroCrossing" method
:param img: Input image
:return: Edge matrix
"""
def edgeDetectionZeroCrossingLOG(img:np.ndarray)->(np.ndarray):
    # smooth the image with a Gaussian filter:
    img = img.astype(float)
    smooth_img = cv2.GaussianBlur(img, (3, 3), 0)
    # convolve the smoothed image with the Laplacian filter:
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    lap_img = cv2.filter2D(smooth_img, -1, lap, borderType=cv2.BORDER_REPLICATE)
    edges = np.zeros_like(lap_img)
    R, C = img.shape
    for x in range(1, R - 1):
        for y in range(1, C - 1):
            neighbors = [lap_img[x-1, y], lap_img[x+1, y], lap_img[x, y-1], lap_img[x, y+1], lap_img[x-1, y+1], lap_img[x-1, y-1], lap_img[x+1, y+1]]
            # {-,0,+}/{+,0,-}:
            if 0 <= lap_img[x, y] < 0.0001:
                if (lap_img[x-1, y] < 0 < lap_img[x+1, y]) or (lap_img[x+1, y] < 0 < lap_img[x-1, y]) or (lap_img[x, y-1] < 0 < lap_img[x, y+1]) or (lap_img[x, y+1] < 0 < lap_img[x, y-1]):
                    edges[x, y] = 255
            else:
                # {-,+}/{+,-}:
                if lap_img[x, y] < 0:
                    for z in neighbors:
                        if z > 0:
                            edges[x, y] = 255
                if lap_img[x, y] > 0:
                    for z in neighbors:
                        if z < 0:
                            edges[x, y] = 255
    for x in range(edges.shape[0]):
        for y in range(edges.shape[1]):
            if edges[x, y] == 255:
                edges[x, y] = 0
            else:
                edges[x, y] = 255
    return edges
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: :return: Edge matrix
    """
def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float)-> (np.ndarray, np.ndarray):
    #Gaussian blur
    smooth_img = cv2.GaussianBlur(img, (3, 3), 0)
    # Compute the direction, magnitude, IX, and Iy:
    directions, magnitude, Ix, Iy = convDerivative(smooth_img)
    # To degrees
    angle = np.degrees(directions)
    magnitude = (magnitude/magnitude.max())*255
    # non maximum suppression:
    edges = np.zeros_like(magnitude)
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = r = magnitude[i, j]
            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            # angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
                # angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            # angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                edges[i, j] = magnitude[i, j]
            else:
                edges[i, j] = 0

    # Hysteresis:
    thrs_1 = thrs_1 * 255
    thrs_2 = thrs_2 * 255
    strong = np.zeros_like(edges)
    for x in range(M):
        for y in range(N):
            if edges[x, y] > thrs_1:
                strong[x, y] = edges[x, y]

    for x in range(M):
        for y in range(N):
            if edges[x, y] <= thrs_2:
                edges[x, y] = 0
            if thrs_2 < edges[x, y] <= thrs_1:
                if strong[x - 1, y] == strong[x + 1, y] == strong[x, y - 1] == strong[x, y + 1] == strong[x - 1, y - 1] == strong[x + 1, y + 1] == strong[x + 1, y - 1] == strong[x - 1, y + 1] == 0:
                    edges[x, y] = 0

    cv_ans = cv2.Canny(img, thrs_2, thrs_1)
    return cv_ans, edges
    """
Detecting edges usint "Canny Edge" method
:param img: Input image
:param thrs_1: T1
:param thrs_2: T2
:return: opencv solution, my implementation
"""

def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param I: Input image
    :param minRadius: Minimum circle radius
    :param maxRadius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """
    cv_ans = cv2.Canny(img, 100, 200)
    edges = []

    for x in range(cv_ans.shape[0]):
        for y in range(cv_ans.shape[1]):
            if cv_ans[x, y] == 255:
                edges.append((x, y))

    # Thresh - 47% need to be edges
    # Count - 100 points
    thresh = 0.47
    mone = 100

    points = []
    for r in range(min_radius, max_radius + 1):
        for t in range(mone):
            # Using trigo
            x = int(r * cos(2 * pi * t / mone))
            y = int(r * sin(2 * pi * t / mone))
            points.append((x, y, r))

    # temp = Loaction : counter
    temp = {}
    for x, y in edges:
        for x0, y0, r in points:
            b = x - x0
            a = y - y0
            count = temp.get((a, b, r))
            if count is None:
                count = 0
            temp[(a, b, r)] = count + 1

    # Check the temp circles and add to the final circles
    circles = []
    sorted_temp = sorted(temp.items(), key=lambda i: -i[1])
    for circle, counter in sorted_temp:
        x, y, r = circle
        # once a circle has been selected, we reject all the circles whose center is inside that circle
        if counter / mone >= thresh and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            print(counter / mone, x, y, r)
            circles.append((x, y, r))

    return circles

def draw(img: np.ndarray, ans: list):
    fig, ax = plt.subplots()
    for x, y, r in ans:
        circle = plt.Circle((x, y), r, color='black', fill=False)
        center = plt.Circle((x, y), 0.5, color='b', )
        ax.add_patch(circle)
        ax.add_patch(center)
    ax.imshow(img)
    plt.show()
