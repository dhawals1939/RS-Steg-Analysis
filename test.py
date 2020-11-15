import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pylab as pl
import random

class LSB_steg_RS(object):

    def __init__(self, img_, size=8):
        self.img_ = img_  # origin image
        self.size = size # SIZE*SIZE image block
        self.height = img_.shape[0]
        self.width = img_.shape[1]
        self.color = img_.shape[2]

    def Zigzag(self, img):
        """
        Convert the two-dimension matrix into one-dimension using Zigzag method
        :param img: the source bmp image pixel two-dimension matrix
        :return: one-dimension matrix of Zigzag
        """
        tmp = np.zeros((self.size, self.size))
        i = 0
        j = 0
        for x in range(self.size): # Zigzag
            for y in range(self.size):
                tmp[x][y] = img[i][j]
                if((i==self.size-1 or i == 0) and j%2==0):
                    j += 1
                    continue
                if((j==0 or j==self.size-1) and i%2==1):
                    i += 1
                    continue
                if((i+j)%2==0):
                    i -= 1
                    j += 1
                elif((i+j)%2==1):
                    i += 1
                    j -= 1
        tmp = tmp.reshape(1,64)
        return tmp[0]

    def Pixel_correlation(self, img_onedimen):
        """
        Calculate the image block pixel(8*8) correlation
        :param img_onedimen: one-dimension matrix of Zigzag
        :return: pixel correlation of image block
        """
        px_correlation = 0
        for i in range(len(img_onedimen)-1):
            if (img_onedimen[i + 1] > img_onedimen[i]):
                px_correlation += img_onedimen[i + 1] - img_onedimen[i]
            else:
                px_correlation += img_onedimen[i] - img_onedimen[i + 1]
        return px_correlation

    def F0(self, pixel):
        """
        F0 turn,namely keep the original pixel
        :param pixel: image pixel
        :return: pixel
        """
        return pixel

    def F1(self, pixel):
        """
        F1 turn,namely 0-1, 2-3, ..., 254-255
        :param pixel: image pixel
        :return: pixel
        """
        if (pixel % 2 == 0):  #0/2/4.../254 upto 1/3/5/.../255
            pixel = pixel + 1
        elif (pixel % 2 == 1): #1/3/5/.../255 downto 0/2/4/.../254
            pixel = pixel - 1
        return pixel

    def F_1(self, pixel):
        """
        F_1 turn,namely -1-0, 1-2, ..., 255-256
        :param pixel: image pixel
        :return: pixel
        """
        if (pixel % 2 == 0): #0/2/4.../254 downto -1/1/3/.../253
            pixel = pixel - 1
        elif (pixel % 2 == 1): #1/3/5.../255 upto 2/4/6/.../256
            pixel = pixel + 1
        return pixel

    def Random_num(self, turn_type):
        """
        Generate random number to decide the F0 or F1,F0 or F_1
        :param turn_type: int number -1 or 1, -1: F0 and F_1, 1: F0 and F1
        :return: one-dimension random number
        """
        SIZE = self.size*self.size
        random_num = np.zeros(SIZE)
        if (turn_type == 1):
            for i in range(SIZE):
                random_num[i] = random.randint(0, 1)
        elif (turn_type == -1):
            for i in range(SIZE):
                random_num[i] = random.randint(-1, 0)
        return random_num

    def F_101RS(self, ftype, img_block):
        """
        RS steganalysis, carry out F0 and F1 or F0 and F_1 turn, calculate pixel correlation
        and then compare them
        :param ftype: F0 and F1 or F0 and F_1
        :return: R,S
        """
        random_num = self.Random_num(ftype)
        after_turn = np.zeros((self.size, self.size))  # restore image block pixel after turn
        R = 0
        S = 0
        one_dimensionimg1 = self.Zigzag(img_block)
        # print one_dimensionimg1
        before_turnRs = self.Pixel_correlation(one_dimensionimg1)  # calculate image block's pixel
        # correlation before turning
        k = 0
        for i in range(self.size):
            for j in range(self.size):
                if (random_num[k] == 0):  # F0 turn
                    after_turn[i][j] = self.F0(img_block[i][j])
                elif (random_num[k] == 1):  # F1 turn
                    after_turn[i][j] = self.F1(img_block[i][j])
                elif (random_num[k] == -1):  # F_1 turn
                    after_turn[i][j] = self.F_1(img_block[i][j])
                k = k + 1
        one_dimensionimg2 = self.Zigzag(after_turn)  # Zigzag
        after_turnRs = self.Pixel_correlation(one_dimensionimg2)  # calculate image block's pixel
        # correlation after turning
        if (before_turnRs < after_turnRs):
            R = R + 1
        elif (before_turnRs > after_turnRs):
            S = S + 1

        return R, S

    def RS(self, img_block):
        """
        RS steganalysis, carry out (F0, F1) and (F0, F_1) turn, calculate pixel correlation
        and then compare them
        :param img_block: 8*8 two-dimension img pixel block
        :return: Rm,Sm,R_m,S_m
        """
        Rm, Sm = self.F_101RS(1, img_block)
        R_m, S_m = self.F_101RS(-1, img_block)

        return Rm, Sm, R_m, S_m

    def LSB(self, img, rate):
        """
        LSB steganalysis
        :param img: image pixel matrix (512*512)
        :param rate: 0~1
        :return: image pixel matrix (512*512) after steganalysis
        """
        s = int(self.height * self.width * rate)
        secret = np.zeros(s) # secret information

        for i in range(s):
            secret[i] = random.randint(0, 1)
        k = 0
        for i in range(self.height):
            for j in range(self.width):
                if (k < s):
                    if (secret[k] == 1 and img[i][j] % 2 == 0):  # even embed 1, add 1
                        img[i][j] = img[i][j] + 1
                        k += 1
                    elif (secret[k] == 1 and img[i][j] % 2 == 1):  # odd embed 1, no change
                        img[i][j] = img[i][j] + 0
                        k += 1
                    elif (secret[k] == 0 and img[i][j] % 2 == 0):  # even embed 0, no change
                        img[i][j] = img[i][j] + 0
                        k += 1
                    elif (secret[k] == 0 and img[i][j] % 2 == 1):  # odd embed 0, sub 1
                        img[i][j] = img[i][j] - 1
                        k += 1
                    else:
                        pass
        return img


    def imgTotal_correlation(self, imgR):
        """
        Calculate the total pixel_correlation of image
        :param img: image
        :param imgR: image matrix of R in RGB
        :return: Rm,Sm,R_m,S_m
        """
        tmp = np.zeros((self.size,self.size))

        row = int(self.height / self.size)  # row/self.size
        col = int(self.width / self.size)  # column/self.size
        result = {'Rm':0,'Sm':0,'R_m':0,'S_m':0}

        for i in range(row):
            for j in range(col):
                x = 0
                for r in range(i*self.size,(i+1)*self.size):
                    y = 0
                    for c in range(j*self.size,(j+1)*self.size):
                        tmp[x][y] = imgR[r][c]
                        y = y + 1
                    x = x + 1
                Rm, Sm, R_m, S_m = self.RS(tmp)  # RS steganalysis
                result['Rm'] += Rm
                result['Sm'] += Sm
                result['R_m'] += R_m
                result['S_m'] += S_m

        return result['Rm'], result['Sm'], result['R_m'], result['S_m']

if __name__ == '__main__':
    import sys
    # reload(sys)
    # sys.setdefaultencoding('gbk')

    # img = mpimg.imread('lena.bmp')

    import cv2
    img = cv2.imread('D:\shiva\IIITH\sem3\DIP\Project\project-photons\images\kyoto.bmp')
    img = cv2.resize(img, (623, 676), interpolation = cv2.INTER_AREA)

    lsb_rs = LSB_steg_RS(img,8)
    plt.imshow(img)  # show img
    #print img
    # plt.imshow(img_1,cmap='gray')
    plt.axis('off')  # no axis
    plt.show()
    print( img.shape)

    # restore img into array
    img_1 = img[:, :, 0]
    print( img_1)
    imgR = np.array(img_1)

    #before LSB steg, we calculate the correlation of img
    Rm, Sm, R_m, S_m = lsb_rs.imgTotal_correlation(imgR)
    print ("Rm = %d, Sm = %d, R_m = %d, S_m = %d"%(Rm, Sm, R_m, S_m))
    # after LSB steg, we calculate the correlation of img
    rate = 0.1
    rm = np.zeros(11)
    sm = np.zeros(11)
    r_m = np.zeros(11)
    s_m = np.zeros(11)
    index = 0
    rm[index] = Rm
    sm[index] = Sm
    r_m[index] = R_m
    s_m[index] = S_m
    imgshow = np.zeros((512,512,3),np.uint8)

    while (rate <= 1.0):
        imgR = lsb_rs.LSB(imgR, rate)
        #print imgR
        #imgR = lsb_rs.LSB_improve(imgR, rate)
        # imgshow[:, :, 0] = imgR
        # imgshow[:, :, 1] = img[:, :, 1]
        # imgshow[:, :, 2] = img[:, :, 2]
        # #imgshow[:, :, 3] = img[:, :, 3]
        # plt.imshow(imgshow)  # show img
        # # # plt.imshow(img_1,cmap='gray')
        # plt.axis('off')  # no axis
        # plt.show()

        Rm, Sm, R_m, S_m = lsb_rs.imgTotal_correlation(imgR)
        print ("Rm = %d, Sm = %d, R_m = %d, S_m = %d" % (Rm, Sm, R_m, S_m))
        rate += 0.1
        index += 1
        rm[index] = Rm
        sm[index] = Sm
        r_m[index] = R_m
        s_m[index] = S_m

    rate = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pl.plot(rate, rm, label='Rm')
    pl.plot(rate, sm, label='Sm')
    pl.plot(rate, r_m, label='R_m')
    pl.plot(rate, s_m, label='S_m')
    plt.legend(['Rm', 'Sm', 'R_m', 'S_m'], loc='upper right')
    plt.savefig("result1.png")
    pl.show()

    
    # def T_score(self, x, y, img):
    #     """
    #     Calculate (x,y)'s all around pixel sum - 9*(x,y) pixel
    #     :param x: row
    #     :param y: column
    #     :param img: image two-dimension matrix
    #     :return: pixel_all-9*pixel
    #     """
    #     pixel = img[x][y]
    #     pixel_all = 0
    #     if x-1>=0 and y-1>=0:
    #         pixel_all += img[x-1][y-1]
    #     if x-1>=0:
    #         pixel_all += img[x-1][y]
    #     if x-1>=0 and y+1<self.width:
    #         pixel_all += img[x-1][y+1]
    #     if y-1>=0:
    #         pixel_all += img[x][y-1]
    #     if y+1<self.width:
    #         pixel_all += img[x][y+1]
    #     if x+1<self.height and y-1>=0:
    #         pixel_all += img[x+1][y-1]
    #     if x+1<self.height:
    #         pixel_all += img[x+1][y]
    #     if x+1<self.height and y+1 <self.width:
    #         pixel_all += img[x+1][y+1]

    #     return (pixel_all-9*pixel)

    # def LSB_improve(self, img, rate):
    #     """
    #     Improved LSB steganalysis
    #     :param img: image pixel matrix (512*512)
    #     :param rate: 0~1
    #     :return: image pixel matrix (512*512) after steganalysis
    #     """
    #     s = int(self.height * self.width * rate)
    #     secret = np.zeros(s) # secret information

    #     for i in range(s):
    #         secret[i] = random.randint(0, 1)
    #     k = 0
    #     for i in range(self.height):
    #         for j in range(self.width):
    #             if (k < s):
    #                 if (secret[k] == 1 and img[i][j] % 2 == 0):  # even embed 1, add 1/sub 1
    #                     if (img[i][j]==255): img[i][j] = img[i][j] - 1
    #                     elif (img[i][j]==0): img[i][j] = img[i][j] + 1
    #                     elif (self.T_score(i,j,img)<=0): img[i][j] = img[i][j] - 1
    #                     elif (self.T_score(i,j,img)>0): img[i][j] = img[i][j] + 1
    #                     k += 1
    #                 elif (secret[k] == 1 and img[i][j] % 2 == 1):  # odd embed 1, no change
    #                     img[i][j] = img[i][j] + 0
    #                     k += 1
    #                 elif (secret[k] == 0 and img[i][j] % 2 == 0):  # even embed 0, no change
    #                     img[i][j] = img[i][j] + 0
    #                     k += 1
    #                 elif (secret[k] == 0 and img[i][j] % 2 == 1):  # odd embed 0, sub 1/add 1
    #                     if (img[i][j]==255): img[i][j] = img[i][j] - 1
    #                     elif (img[i][j]==0): img[i][j] = img[i][j] + 1
    #                     elif (self.T_score(i,j,img)<=0): img[i][j] = img[i][j] - 1
    #                     elif (self.T_score(i,j,img)>0): img[i][j] = img[i][j] + 1
    #                     k += 1
    #                 else:
    #                     pass
    #     return img