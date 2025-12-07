# reference https://juejin.cn/post/6948595057624612900
import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
 
def classify_gray_hist(image1,image2,size = (256,256)): 
 image1 = cv2.resize(image1,size) 
 image2 = cv2.resize(image2,size) 
 hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0]) 
 hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0]) 
 plt.plot(range(256),hist1,'r') 
 plt.plot(range(256),hist2,'b') 
#  plt.show() 
 degree = 0 
 for i in range(len(hist1)): 
    if hist1[i] != hist2[i]: 
        degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i])) 
    else: 
        degree = degree + 1 
    
 degree = degree/len(hist1) 
 return degree 
 
def calculate(image1,image2): 
 hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0]) 
 hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0]) 
 degree = 0 
 for i in range(len(hist1)): 
    if hist1[i] != hist2[i]: 
        degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i])) 
    else: 
        degree = degree + 1

 degree = degree/len(hist1)

 if degree == 1:
     return degree

 return degree[0]
 
def classify_hist_with_split(image1,image2,size = (256,256)): 
 image1 = cv2.resize(image1,size) 
 image2 = cv2.resize(image2,size) 
 sub_image1 = cv2.split(image1) 
 sub_image2 = cv2.split(image2) 
 sub_data = 0 
 for im1,im2 in zip(sub_image1,sub_image2): 
    sub_data += calculate(im1,im2)
    
 sub_data = sub_data/3 
 return sub_data 
 
def classify_aHash(image1,image2): 
 image1 = cv2.resize(image1,(8,8)) 
 image2 = cv2.resize(image2,(8,8)) 
 gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY) 
 gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY) 
 hash1 = getHash(gray1) 
 hash2 = getHash(gray2) 
 return Hamming_distance(hash1,hash2) 
 
def classify_pHash(image1,image2): 
 image1 = cv2.resize(image1,(32,32)) 
 image2 = cv2.resize(image2,(32,32)) 
 gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY) 
 gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY) 
 dct1 = cv2.dct(np.float32(gray1)) 
 dct2 = cv2.dct(np.float32(gray2)) 
 dct1_roi = dct1[0:8,0:8] 
 dct2_roi = dct2[0:8,0:8] 
 hash1 = getHash(dct1_roi) 
 hash2 = getHash(dct2_roi) 
 return Hamming_distance(hash1,hash2) 
 
def getHash(image): 
 avreage = np.mean(image) 
 hash = [] 
 for i in range(image.shape[0]): 
    for j in range(image.shape[1]): 
        if image[i,j] > avreage: 
            hash.append(1) 
        else: 
            hash.append(0) 
 return hash 
 
 
def Hamming_distance(hash1,hash2): 
 num = 0 
 for index in range(len(hash1)): 
    if hash1[index] != hash2[index]: 
        num += 1 
 return num 
 
def _ssim_similarity(image1, image2):
 image1 = cv2.resize(image1, (256, 256))
 image2 = cv2.resize(image2, (256, 256))
 gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
 gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
 x = gray1.astype(np.float32)
 y = gray2.astype(np.float32)
 mu_x = np.mean(x)
 mu_y = np.mean(y)
 sigma_x2 = np.var(x)
 sigma_y2 = np.var(y)
 sigma_xy = np.mean((x - mu_x) * (y - mu_y))
 C1 = (0.01 * 255) ** 2
 C2 = (0.03 * 255) ** 2
 num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
 den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2)
 if den == 0:
     return 0.0
 s = num / den
 if s < 0:
     s = 0.0
 if s > 1:
     s = 1.0
 return float(s)

def compareImg(img1, img2, metric = 'hist'):
 if metric == 'hist':
     return float(classify_hist_with_split(img1, img2))
 elif metric == 'ahash':
     d = float(classify_aHash(img1, img2))
     return 1.0 - d / 64.0
 elif metric == 'phash':
     d = float(classify_pHash(img1, img2))
     return 1.0 - d / 64.0
 elif metric == 'ssim':
     return _ssim_similarity(img1, img2)
 else:
     return float(classify_hist_with_split(img1, img2))
 
if __name__ == '__main__': 
 img1 = cv2.imread('./data/frame8.jpg') 
#  cv2.imshow('img1',img1) 
 img2 = cv2.imread('./data/frame9.jpg') 
#  cv2.imshow('img2',img2) 
#  degree = classify_gray_hist(img1,img2) 
 degree = classify_hist_with_split(img1,img2) 
#  degree = classify_aHash(img1,img2) 
#  degree = classify_pHash(img1,img2) 

 cv2.destroyAllWindows()
 exit(0)
