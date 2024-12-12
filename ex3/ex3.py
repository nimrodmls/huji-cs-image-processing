import PIL.ImageFilter
import PIL.Image
import numpy as np
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter

gaussian_kernel_3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

def gaussian_kernel(size, sigma=1):
    """
    :param size: The size of the kernel
    :param sigma: The standard deviation of the gaussian
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

GAUSSIAN_SIZE = 2

def gaussian_blur(img, size):
    """
    :param img: A single channel image array
    :param size: The size of the kernel
    """
    # The fill is defaulted to zero
    return convolve(img, gaussian_kernel(size), 'same')
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

def expand_image(img, gaussian_size):
    """
    Expanding the image by a factor of 2, with zero padding,
    and application of a gaussian blur
    """
    upsample_factor = 2

    return rgb_gaussian_blur(upsample(img, upsample_factor), gaussian_size)

def reduce_image(img, gaussian_size):
    """
    Reducing the image by a factor of 2, with application of a gaussian blur
    """
    downsample_factor = 2

    return downsample(rgb_gaussian_blur(img, gaussian_size), downsample_factor)

def gaussian_pyramid(img, levels, gaussian_size):
    """
    Creating a gaussian pyramid of the given image
    Returns a list of images, where the first image is the original image,
    and each subsequent image is a reduced version of the previous
    """
    pyramid = [img]
    for lvl in range(levels - 1):
        img = reduce_image(img, gaussian_size)
        pyramid.append(img)

    return pyramid

def laplacian_pyramid(img, levels, gaussian_size):
    """
    Creating a laplacian pyramid of the given image
    Returns a list of images, where the last image is the smallest,
    and each subsequent image is the difference between the previous 
    and the expanded version of the previous
    """
    pyramid = []
    for lvl in range(levels - 1):
        reduced = reduce_image(img, gaussian_size)
        expanded = expand_image(reduced, gaussian_size)
        # Trim the expanded image to the size of the reduced image
        expanded = expanded[:img.shape[0], :img.shape[1]]
        diff = img - expanded
        pyramid.append(diff)
        img = reduced

    pyramid.append(img)
    return pyramid

def reconstruct_image(pyramid, gaussian_size):
    """
    """
    img = pyramid[-1]
    for lvl in range(len(pyramid) - 2, -1, -1):
        expanded = expand_image(img, gaussian_size)
        # Trim the expanded image to the size of the reduced image
        expanded = expanded[:pyramid[lvl].shape[0], :pyramid[lvl].shape[1]]
        img = expanded + pyramid[lvl]

    return img

def image_blend(img1, img2, mask):
    """
    Blending two images using the given mask
    Implemented with Gaussian & Laplacian Pyramids
    """
    pyramid_levels = 5

    img1_laplacian = laplacian_pyramid(img1, pyramid_levels, 2)
    img2_laplacian = laplacian_pyramid(img2, pyramid_levels, 2)
    mask_gaussian = gaussian_pyramid(mask, pyramid_levels, 10)

    # Blending the images, as taught in class
    blended_pyramid = []
    for i in range(pyramid_levels):
        blended_pyramid.append(
            (mask_gaussian[i] * img1_laplacian[i]) + ((1 - mask_gaussian[i]) * img2_laplacian[i]))
        
    return reconstruct_image(blended_pyramid, 2)

def read_image(path):
    """
    Reads an image and returns it as a numpy array, normalized to [0, 1]
    """
    return np.array(PIL.Image.open(path)) / 255

def save_image(img, path):
    """
    Saves an image to the given path, assuming the image is normalized to [0, 1]
    """
    img = np.clip(img, 0, 1)
    PIL.Image.fromarray((img * 255).astype(np.uint8)).save(path)

def main():
    img1 = read_image('doge.png')
    img2 = read_image('musk.png')
    mask = read_image('mask.png')

    image_blend_result = image_blend(img1, img2, mask)
    save_image(image_blend_result, 'blended_image.png')


if __name__ == "__main__":
    main()