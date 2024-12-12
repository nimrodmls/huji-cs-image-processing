import PIL.ImageFilter
import PIL.Image
import numpy as np
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter

gaussian_kernel_3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

# def gaussian_kernel(size, sigma=1):
#     """
#     :param size: The size of the kernel
#     :param sigma: The standard deviation of the gaussian
#     """
#     ax = np.arange(-size // 2 + 1., size // 2 + 1.)
#     xx, yy = np.meshgrid(ax, ax)
#     kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
#     return kernel / np.sum(kernel)

def gaussian_blur(img, size):
    """
    :param img: A single channel image array
    :param size: The size of the kernel
    """
    # The fill is defaulted to zero
    return convolve(img, gaussian_kernel_3, 'same')
    # return gaussian_filter(img, size, mode='constant')

def rgb_gaussian_blur(img, size):
    """
    :param img: A 3 channel image array
    :param size: The size of the kernel
    """
    return np.stack([gaussian_blur(img[:, :, i], size) for i in range(3)], axis=-1)

def downsample(img, factor):
    """
    Scaling down the image by the given factor
    """
    return img[::factor, ::factor]

def upsample(img, factor):
    """
    Scaling up the image by the given factor, with zero padding
    """
    upsampled_img = np.zeros((img.shape[0] * factor, img.shape[1] * factor, img.shape[2]))
    upsampled_img[::factor, ::factor] = img

    return upsampled_img

def expand_image(img):
    """
    Expanding the image by a factor of 2, with zero padding,
    and application of a gaussian blur
    """
    upsample_factor = 2
    gaussian_size = 1

    return rgb_gaussian_blur(upsample(img, upsample_factor), gaussian_size)

def reduce_image(img):
    """
    Reducing the image by a factor of 2, with application of a gaussian blur
    """
    downsample_factor = 2
    gaussian_size = 1

    return downsample(rgb_gaussian_blur(img, gaussian_size), downsample_factor)

def read_image(path):
    """
    """
    return np.array(PIL.Image.open(path)) / 255

def save_image(img, path):
    """
    """
    img = np.clip(img, 0, 1)
    PIL.Image.fromarray((img * 255).astype(np.uint8)).save(path)

def main():
    img = read_image("doge.jpg")
    reduced_img = reduce_image(img)
    save_image(reduced_img, "doge_reduced.jpg")
    expanded_img = expand_image(reduced_img)
    save_image(expanded_img, "doge_expanded.jpg")


if __name__ == "__main__":
    main()