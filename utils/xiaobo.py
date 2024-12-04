import numpy as np
from scipy.__config__ import show
import torch
import pywt  
import matplotlib.pyplot as plt
import cv2
def xiaobo(model,step):
    N = 1000
    tspace = np.linspace(0., 0.6, N )
    xspace = np.linspace(-1.,1., N )
    T, X = np.meshgrid(tspace, xspace)
    Xgrid = np.vstack([T.flatten(),X.flatten()]).T
    upred = model(torch.tensor(Xgrid,dtype=torch.float32).cuda())
    U = upred.cpu().detach().numpy().reshape(N,N)
    img = U


    #小波变换
    coeffs = pywt.dwt2(img, 'db1')
    p,(q,r,s)=coeffs
  
  
    
    '''
    filtered_coeffs_l=(p,(np.zeros_like(q),np.zeros_like(r),np.zeros_like(s)))
    filtered_img_l = pywt.idwt2(filtered_coeffs_l, 'db1') 
    iimg_l = filtered_img_l.reshape(-1)
    topindex_l=np.argpartition(iimg_l, -100000)
    X_l=Xgrid[topindex_l]
    X_l=X_l[-100000:]
    '''
    filtered_coeffs_h=(np.zeros_like(p),(q,r,s))
    filtered_img_h = pywt.idwt2(filtered_coeffs_h, 'db1') 
    iimg_h = filtered_img_h.reshape(-1)
    filtered_img_h=(filtered_img_h-np.min(filtered_img_h))/(np.max(filtered_img_h)-np.min(filtered_img_h))*255
    plt.hist(filtered_img_h,bins=10)
    plt.ylim(0,550)
    plt.show()
    dst = cv2.equalizeHist(filtered_img_h.astype(np.uint8))
    plt.hist(dst,bins=10)
    plt.ylim(0,550)
    plt.show()
    pc=dst.reshape(-1)
    pc=pc/np.sum(pc)

    if step <5000:
        n=0
    if 5000<=step<10000:
        n=10
    if 10000<=step<15000:
        n=20
    if step>=15000:
        n=30
    '''
    if step <10000:
        n=0
    if 10000<=step <20000:
        n=5
    if 20000<=step <30000:
        n=10
    if 30000<= step<40000:
        n=15
    if step>=40000:
        n=20
    '''
    n=step
    pc=pc**n
    pc=pc/np.sum(pc)
    seed_indxs = np.random.choice(
            len(pc),
            2000,
            p=pc,
            replace=False
        )
    point=Xgrid[seed_indxs]
    #plt.scatter(point[:,0],point[:,1])
    #plt.show()
    """
    topindex_h=np.argpartition(iimg_h, -100000)
    va=iimg_h[topindex_h]
    va=va[-100000:]
    X_h=Xgrid[topindex_h]
    X_h=X_h[-100000:]
    """
    return point
def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def ran(values,points,n):
    data_min = np.min(values)  
    data_max = np.max(values)   
    p = (values - data_min) / (data_max - data_min) 
    p = p/np.sum(p) 
    print(data_min,data_max)
    seed_indxs = np.random.choice(
            len(values),
            n,
            p=p,
            replace=False
        )
    r=points[seed_indxs]
    return r


def fuse(points, d):
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count+=1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append([point[0], point[1]])
    return ret

def fusample(model,step=0):
    a_h=xiaobo(model,step)
    #b_h=fuse(a_h,0.035)
    b=a_h
    r=torch.tensor(b,dtype=torch.float32).cuda()
    rnp=np.array(b,dtype=np.float32)
    return r,rnp