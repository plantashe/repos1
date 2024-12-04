import numpy as np
import torch
def fu(model):
    N = 1000
    tspace = np.linspace(0., 0.5, N )
    xspace = np.linspace(-1.,1., N )
    T, X = np.meshgrid(tspace, xspace)
    Xgrid = np.vstack([T.flatten(),X.flatten()]).T
    upred = model(torch.tensor(Xgrid,dtype=torch.float32).cuda())
    U = upred.cpu().detach().numpy().reshape(N,N)
    img = U
    #傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    #设置高通滤波器
    rows, cols = img.shape
    crow,ccol = int(rows/2), int(cols/2)
    fshift[crow-20:crow+21, ccol-20:ccol+21] = 0
    #傅里叶逆变换
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    iimg[:30,:]=0
    iimg[-30:,:]=0
    iimg[:,:30]=0
    iimg[:,-30:]=0
    iimg = iimg.reshape(-1)
    topindex=np.argpartition(iimg, -100000)
    X=Xgrid[topindex]
    X=X[-100000:]
    return X

def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

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

def fusample(model):
    a=fu(model)
    b=fuse(a,0.05)
    r=torch.tensor(b,dtype=torch.float32).cuda()
    rnp=np.array(b,dtype=np.float32)
    return r,rnp