import numpy as np
import random
import string

# research paper discrimination function
# def discrimination_function(np_img_window):
#     if len(np_img_window.shape) == 3: # for RGB
#         # np_img_window_0, np_img_window_1, np_img_window_2 = np_img_window[:,:,0].flatten(), np_img_window[:,:,1].flatten(), np_img_window[:,:,2].flatten()

#         # channelSum0 = np.sum(np.abs(np_img_window_0[:-1] - np_img_window_0[1:])) 
#         # channelSum1 = np.sum(np.abs(np_img_window_1[:-1] - np_img_window_1[1:]))
#         # channelSum2 = np.sum(np.abs(np_img_window_2[:-1] - np_img_window_2[1:]))

#         # return (channelSum0 + channelSum1 + channelSum2)
#         pass
    
#     elif len(np_img_window.shape) == 2: # for grayscale
#         np_img_window = np_img_window.flatten()
#         return np.sum(np.abs(np_img_window[:-1] - np_img_window[1:]))
    
#     else:
#         raise Exception("Error: Invlaid shape of image in discrimination_function")
    # Extra subtractions across channels -- Is this a problem?

def discrimination_function(np_img_window):
    np_img_window = np.concatenate([np.diagonal(np_img_window[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-np_img_window.shape[0], np_img_window.shape[0])]) # zigzag scan
    return np.sum(np.abs(np_img_window[:-1] - np_img_window[1:]))

def support_f_1(np_img_window):
    np_img_window = np.copy(np_img_window)
    even_values = np_img_window % 2 == 0
    np_img_window[even_values] += 1
    np_img_window[np.logical_not(even_values)] -= 1
    return np_img_window

def flipping_operation(np_img_window, np_mask):
    f_1 = lambda x: support_f_1(x)
    f_0 = lambda x: np.copy(x)
    f_neg1 = lambda x: support_f_1(x+1)-1
    np_result = np.empty(np_img_window.shape)
    dict_flip = {-1:f_neg1, 0:f_0, 1:f_1}
    for i in [-1, 0, 1]:
        temp_indices_x, temp_indices_y = np.where(np_mask == i)
        np_result[temp_indices_x, temp_indices_y] = dict_flip[i](np_img_window[temp_indices_x, temp_indices_y])
    return np_result

def calculate_count_groups(np_img, np_mask):
    count_reg, count_sing, count_unusable = 0, 0, 0

    for ih in range(0, np_img.shape[0], np_mask.shape[0]):
        for iw in range(0, np_img.shape[1], np_mask.shape[1]):
            np_img_window = np_img[ih: ih+np_mask.shape[0], iw: iw+np_mask.shape[1]]    # this is one group
            flipped_output = flipping_operation(np_img_window, np_mask)
            
            discrimination_img_window = discrimination_function(np_img_window)
            discrimination_flipped_output = discrimination_function(flipped_output)

            if discrimination_flipped_output > discrimination_img_window: 
                count_reg += 1
            elif discrimination_flipped_output < discrimination_img_window:
                count_sing += 1
            else:
                count_unusable += 1

    totalGroups = count_reg + count_sing + count_unusable
    return count_reg/totalGroups, count_sing/totalGroups
    # return count_reg, count_sing

def randomString(n):
    chars = string.ascii_lowercase
    return ''.join(random.choice(chars) for _ in range(n))

def simpleLSBFlipper(image, n):
    flattenedImg = image.flatten()
    flipped = flattenedImg[0:n]^1       # 0 XOR 1 = 1 ; 1 XOR 1 = 0
    flattenedImg[0:n] = flipped
    return flattenedImg.reshape(image.shape)

def scattered_lsb_flipping(np_img, percent):
    np_result = np.copy(np_img)
    no_pixels_to_change = int(np.round(percent * np.prod(np_img.shape)))
    random_indices_c = np.random.randint(low = 0, high = np_img.shape[1], size = no_pixels_to_change)
    random_indices_r = np.random.randint(low = 0, high = np_img.shape[0], size = no_pixels_to_change)
    np_result[random_indices_r, random_indices_c] = np.bitwise_xor(np_result[random_indices_r, random_indices_c], 1)
    return np_result

def RS_quadratic_solver(half_LSBs_flipped, all_LSBs_flipped):

    RM_p2 = half_LSBs_flipped[0]
    SM_p2 = half_LSBs_flipped[1]
    R_minus_M_p2 = half_LSBs_flipped[2]
    S_minus_M_p2 = half_LSBs_flipped[3]

    RM_1_minus_p2 = all_LSBs_flipped[0]
    SM_1_minus_p2 = all_LSBs_flipped[1]
    R_minus_M_1_minus_p2 = all_LSBs_flipped[2]
    S_minus_M_1_minus_p2 = all_LSBs_flipped[3]

    d0        = RM_p2                -  SM_p2
    d1        = RM_1_minus_p2        -  SM_1_minus_p2
    d_minus_0 = R_minus_M_p2         -  S_minus_M_p2
    d_minus_1 = R_minus_M_1_minus_p2 -  S_minus_M_1_minus_p2

    a = 2 * (d1 + d0)
    b = (d_minus_0 - d_minus_1 - d1 - 3*d0)
    c = (d0 - d_minus_0)

    coefficients = [a, b, c]

    return np.roots(coefficients)