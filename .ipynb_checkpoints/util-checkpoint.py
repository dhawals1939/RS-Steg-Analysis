import numpy as np
import random
import string


def discrimination_function(img_window: np.array) -> np.array:
    """
    :param img_window: np.array of img window
    :return: discriminated output

    Function: takes a zig-zag scan of img_window then returns
    Sum of abs(X[i] - X[i-1])
    """
    img_window = np.concatenate([
        np.diagonal(img_window[::-1, :], k)[::(2 * (k % 2) - 1)]
        for k in range(1 - img_window.shape[0], img_window.shape[0])
    ])  # zigzag scan
    return np.sum(np.abs(img_window[:-1] - img_window[1:]))


def support_f_1(img_window: np.array) -> np.array:
    """
    :param img_window: img to window to flip
    :return: flipped img_window
    Support for F1 function
    """
    img_window = np.copy(img_window)
    even_values = img_window % 2 == 0
    img_window[even_values] += 1
    img_window[np.logical_not(even_values)] -= 1
    return img_window


def flipping_operation(img_window: np.array, mask: np.array) -> np.array:
    """
    :param img_window: img window for flipping
    :param mask: mask using which what flipping to be performed at that location
    :return: np.array of flipped image based on mask
    Function performs flipping operation based on the masked passed in as input
    """

    def f_1(x: np.array) -> np.array: return support_f_1(x)

    def f_0(x: np.array) -> np.array: return np.copy(x)

    def f_neg1(x: np.array) -> np.array: return support_f_1(x + 1) - 1

    result = np.empty(img_window.shape)
    dict_flip = {-1: f_neg1, 0: f_0, 1: f_1}
    for i in [-1, 0, 1]:
        temp_indices_x, temp_indices_y = np.where(mask == i)
        result[temp_indices_x, temp_indices_y] = dict_flip[i](img_window[temp_indices_x, temp_indices_y])
    return result


def calculate_count_groups(img: np.array, mask: np.array) -> tuple:
    """
    :param img: input image
    :param mask: mask using which the flipping is done
    :return: tuple of Rm/R-m and Sm/S-m groups are calculated
    Function divides the image into windows of size of mask. Flips it according to the
    window passed checks whether the windows are Regular or Singular based on
    discriminant(flipped)>discriminant(original) --> Regular group else singular
    """
    count_reg, count_sing, count_unusable = 0, 0, 0

    for ih in range(0, img.shape[0], mask.shape[0]):
        for iw in range(0, img.shape[1], mask.shape[1]):
            img_window = img[ih: ih + mask.shape[0], iw: iw + mask.shape[1]]  # this is one group
            flipped_output = flipping_operation(img_window, mask)

            discrimination_img_window = discrimination_function(img_window)
            discrimination_flipped_output = discrimination_function(flipped_output)

            if discrimination_flipped_output > discrimination_img_window:
                count_reg += 1
            elif discrimination_flipped_output < discrimination_img_window:
                count_sing += 1
            else:
                count_unusable += 1

    total_groups = (count_reg + count_sing + count_unusable) / 100  # divided by 100 for calculation in scale of 0-100
    return count_reg / total_groups, count_sing / total_groups


def randomString(n):
    chars = string.ascii_lowercase
    return ''.join(random.choice(chars) for _ in range(n))
s

def scattered_lsb_flipping(img: np.array, percent: float) -> np.array:
    """
    :param img:
    :param percent: percentage of lsb flips need to be done
    :return: resultant image after flipping
    """
    result = np.copy(img)
    no_pixels_to_change = int(np.round(percent * np.prod(img.shape)))
    random_indices_c = np.random.randint(low=0, high=img.shape[1], size=no_pixels_to_change)
    random_indices_r = np.random.randint(low=0, high=img.shape[0], size=no_pixels_to_change)
    result[random_indices_r, random_indices_c] = np.bitwise_xor(result[random_indices_r, random_indices_c], 1)
    return result
