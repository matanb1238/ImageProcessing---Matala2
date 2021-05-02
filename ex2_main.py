from ex2_utils import *
import matplotlib.pyplot as plt
image_path = "beach.jpg"
img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

def ID(id: int):
    print(id)
def conv1Demo():
    sign = np.array([1, 2, 3])
    ker = np.array([1, 1])
    print("Conv1D answer = ", conv1D(sign, ker))

def conv2Demo():
    kernel = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
    ans = conv2D(img, kernel)
    fig2 = plt.figure(2)
    ax1, ax2 = fig2.add_subplot(121), fig2.add_subplot(122)
    ax2.imshow(ans, cmap='gray')
    ax2.set_title("My answer", fontdict=None, loc=None, pad=20, y=None)
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Image", fontdict=None, loc=None, pad=20, y=None)
    fig2.show()
    plt.show()

def derivDemo():
    d, m, y, x = convDerivative(img)
    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(d, cmap='gray')
    ax[0][1].imshow(m, cmap='gray')
    ax[1][0].imshow(x, cmap='gray')
    ax[1][1].imshow(y, cmap='gray')
    ax[0][0].set_title("d", fontdict=None, loc=None, pad=20, y=None)
    ax[0][1].set_title("m", fontdict=None, loc=None, pad=20, y=None)
    ax[1][0].set_title("x", fontdict=None, loc=None, pad=20, y=None)
    ax[1][1].set_title("y", fontdict=None, loc=None, pad=20, y=None)
    plt.show()

def blurDemo():
    image_path2 = "Image.jpeg"
    image = cv2.cvtColor(cv2.imread(image_path2), cv2.COLOR_BGR2GRAY)
    ans1 = blurImage1(image, np.array([3, 3]))
    ans2 = blurImage2(image, np.array([3, 3]))
    fig2 = plt.figure(3)
    ax1, ax2 = fig2.add_subplot(121), fig2.add_subplot(122)
    ax2.imshow(ans1, cmap='gray')
    ax2.set_title("My answer", fontdict=None, loc=None, pad=20, y=None)
    ax1.imshow(ans2, cmap='gray')
    ax1.set_title("Cv answer", fontdict=None, loc=None, pad=20, y=None)
    fig2.show()
    plt.show()

def edgeDemo():
    my_sobel, cv_sobel = edgeDetectionSobel(img, 0.3)
    log = edgeDetectionZeroCrossingLOG(img)
    cv_canny, my_canny = edgeDetectionCanny(img, 0.6, 0.2)
    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(img, cmap='gray')
    ax[0][1].imshow(my_sobel, cmap='gray')
    ax[1][0].imshow(log, cmap='gray')
    ax[1][1].imshow(my_canny, cmap='gray')
    ax[0][0].set_title("Image", fontdict=None, loc=None, pad=20, y=None)
    ax[0][1].set_title("Sobel", fontdict=None, loc=None, pad=20, y=None)
    ax[1][0].set_title("Log", fontdict=None, loc=None, pad=20, y=None)
    ax[1][1].set_title("Canny", fontdict=None, loc=None, pad=20, y=None)
    plt.show()

def houghDemo():
    new_image_path = "pool_balls.jpeg"
    new_img = cv2.cvtColor(cv2.imread(new_image_path), cv2.COLOR_BGR2GRAY)
    circle = houghCircle(new_img, 18, 20)
    draw(new_img, circle)
def main():
    ID(323010835)
    conv1Demo()
    conv2Demo()
    derivDemo()
    blurDemo()
    edgeDemo()
    houghDemo()


if __name__ == '__main__':
    main()
