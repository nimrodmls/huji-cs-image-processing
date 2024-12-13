import PIL.ImageFilter
import PIL.Image
import numpy as np
from scipy.signal import convolve
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def gaussian_kernel(size):
    """
    :param size: The size of the kernel
    :param sigma: The standard deviation of the gaussian
    """
    # Computing the 1D gaussian kernel
    kernel = np.array([1]).astype('int64')
    for i in range(2*size):
        kernel = np.convolve([1, 1], kernel)

    # Transforming to 2D
    x_kern, y_kern = np.meshgrid(kernel, kernel)
    kernel = x_kern * y_kern
    return kernel

def gaussian_blur(img, size, normalization_factor=1):
    """
    :param img: A single channel image array
    :param size: The size of the kernel
    """
    # The fill is defaulted to zero
    kernel = gaussian_kernel(size)
    kernel = kernel / np.sum(kernel) * normalization_factor
    return convolve(img, kernel, mode='same')
    # return gaussian_filter(img, size, mode='constant')

def rgb_gaussian_blur(img, size, normalization_factor=1):
    """
    :param img: A 3 channel image array
    :param size: The size of the kernel
    """
    return np.stack(
        [gaussian_blur(img[:, :, i], size, normalization_factor) for i in range(3)], 
        axis=-1)

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

    return rgb_gaussian_blur(upsample(img, upsample_factor), gaussian_size, normalization_factor=4)

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
    mask_gaussian = gaussian_pyramid(mask, pyramid_levels, 3)

    # Blending the images, as taught in class
    blended_pyramid = []
    for i in range(pyramid_levels):
        blended_pyramid.append(
            (mask_gaussian[i] * img1_laplacian[i]) + ((1 - mask_gaussian[i]) * img2_laplacian[i]))
        
    return reconstruct_image(blended_pyramid, 2)

def show_image_fourier_spectrum(img):
    """
    """
    spectrum = np.abs(fft2(img))
    spectrum = np.log(spectrum + 1)
    plt.imshow(spectrum, cmap='gray')
    plt.show()

def fft_high_pass_filter(img):
    """
    """
    spectrum = fft2(img)
    spectrum[:spectrum.shape[0] * 15 // 16] = 0
    return np.abs(ifft2(spectrum))

def gaussian_high_pass_filter(img):
    """
    """
    kernel = gaussian_kernel(15)
    kernel = kernel / np.sum(kernel)
    return img - convolve(img, kernel, mode='same')

def hybrid_image(img1, img2):
    """
    """
    img1_low = gaussian_blur(img1, 15)
    # img2_high = fft_high_pass_filter(img2)
    img2_high = gaussian_high_pass_filter(img2)
    return img1_low + img2_high
    
def read_image(path):
    """
    Reads an image and returns it as a numpy array, normalized to [0, 1]
    """
    return np.array(PIL.Image.open(path).convert('L')) / 255

def save_image(img, path):
    """
    Saves an image to the given path, assuming the image is normalized to [0, 1]
    """
    img = np.clip(img, 0, 1)
    PIL.Image.fromarray((img * 255).astype(np.uint8)).save(path)

def main():
    # img1 = read_image('doge.png')
    # img2 = read_image('musk.png')
    # mask = read_image('mask.png')

    # image_blend_result = image_blend(img1, img2, mask)
    # save_image(image_blend_result, 'blended_image.png')

    img1 = read_image('trump.jpg')
    img2 = read_image('davegrohl.jpg')
    save_image(hybrid_image(img1, img2), 'trump_high_pass.png')

if __name__ == "__main__":
    main()