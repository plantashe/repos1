from locale import normalize
import logging
import os.path
from random import uniform
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
from utils.models import FullyConnectedNetwork
import utils.fu as fu
import utils.xiaobo as xiaobo
import matplotlib.pyplot as plt
import utils.equations.Burgers as eq
from utils.pde_utils import fwd_gradients
import pywt  
import utils.xiaobo as xiaobo
import cv2
import os
import brewer2mpl
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
@hydra.main(version_base=None, config_path="./conf", config_name="evaluation_conf")
def evaluation_setup(cfg):
    model_conf = cfg["model_conf"]
    equation_conf = cfg["equation_conf"]
    device = torch.device("cuda")
    conf=equation_conf['Burgers']
    #weight_path = "F:/StudyNote/PINN_e/DMIS-main/outputs/2023-11-23/11-48-18/KDV_47000.pth"
    weight_path = "F:/StudyNote/PINN_e/DMIS-main/outputs/2023-12-07/18-57-29/Burgers_18000.pth"
    model_conf["layer"]["layer_n"] = conf["layer_n"]
    model_conf["layer"]["layer_size"] = conf["layer_size"]
    model_conf["dim"]["output_dim"] = conf["output_dim"]
    model = FullyConnectedNetwork(model_conf).to(device)
    model.load_state_dict(torch.load(weight_path))

    def pde_loss(pred, input_tensor):
        df_dt_dx = fwd_gradients(pred, input_tensor)
        df_dt = df_dt_dx[:, 0:1]
        df_dx = df_dt_dx[:, 1:2]
        df_dxx = fwd_gradients(df_dx, input_tensor)[:, 1:2]
        pde_output = df_dt + pred * df_dx - (0.04/torch.pi) * df_dxx
        return pde_output
    def compute_loss_basic_weights(model, data):
        pde_pred = model(data)
        pde_losss = torch.abs(pde_loss(pde_pred, data))
        return pde_losss
    def losstop(model):
        def pde_loss(pred, input_tensor):
            df_dt_dx = fwd_gradients(pred, input_tensor)
            df_dt = df_dt_dx[:, 0:1]
            df_dx = df_dt_dx[:, 1:2]
            df_dxx = fwd_gradients(df_dx, input_tensor)[:, 1:2]
            pde_output = df_dt + pred * df_dx - (0.04/torch.pi) * df_dxx
            return pde_output
        def compute_loss_basic_weights(model, data):
            pde_pred = model(data)
            pde_losss = torch.abs(pde_loss(pde_pred, data))
            return pde_losss
        N = 1000
        tspace = np.linspace(0., 0.6, N )
        xspace = np.linspace(-1.,1., N )
        T, X = np.meshgrid(tspace, xspace)
        Xgrid = np.vstack([T.flatten(),X.flatten()]).T
        data=torch.tensor(Xgrid,dtype=torch.float32,requires_grad=True).cuda()
        loss=compute_loss_basic_weights(model=model,data=data)
        img=loss.cpu().detach().numpy().reshape(N,N)
        coeffs = pywt.dwt2(img, 'db1')
        p,(q,r,s)=coeffs
        filtered_coeffs_h=(np.zeros_like(p),(q,r,s))
        filtered_img_h = pywt.idwt2(filtered_coeffs_h, 'db1') 
        return filtered_img_h,loss
    
    

    N = 1000
    tspace = np.linspace(0., 0.6, N )
    xspace = np.linspace(-1.,1., N )
    ax0=plt.subplot(2,3,1)
    a,b=losstop(model)
    cmap=brewer2mpl.get_map('RdBu', 'diverging', 8, reverse=True).mpl_colormap
    #im = ax0.pcolormesh(tspace, xspace, b.reshape(N,N).cpu().detach().numpy(),shading='auto',cmap=cmap)
    im = ax0.pcolormesh(tspace, xspace,a,shading='auto',cmap=cmap)
    plt.title('(a)loss',y=-0.2)
    cbar =plt.colorbar(im, ax=ax0,location='right')
    cbar.ax.tick_params(labelsize=8)
    ax0.set_aspect(1.0/ax0.get_data_ratio(), adjustable='box')
    ax0.set_xlabel('$t$',loc='left')
    ax0.set_ylabel('$x$',loc='bottom')

    ax1=plt.subplot(2,3,2)
    ps,uniform500=xiaobo.fusample(model,step=0)
    ax1.scatter(uniform500[:,0],uniform500[:,1],s=1,c='salmon')
    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
    ax1.set_xlim([0, 0.6])  
    ax1.set_ylim([-1, 1])
    plt.title('(b)$\\beta=0$',y=-0.2)
    ax1.set_xlabel('$t$',loc='left')
    ax1.set_ylabel('$x$',loc='bottom')
    
   

    ax2=plt.subplot(2,3,3)
    ps,uniform500=xiaobo.fusample(model,step=10)
    ax2.scatter(uniform500[:,0],uniform500[:,1],s=1,c='salmon')
    ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
    ax2.set_xlim([0, 0.6])  
    ax2.set_ylim([-1, 1])
    plt.title('(c)$\\beta=10$',y=-0.2)
    ax2.set_xlabel('$t$',loc='left')
    ax2.set_ylabel('$x$',loc='bottom')
  
    ax3=plt.subplot(2,3,4)
    ps,uniform500=xiaobo.fusample(model,step=20)
    ax3.scatter(uniform500[:,0],uniform500[:,1],s=1,c='salmon')
    ax3.set_xlim([0, 0.6])  
    ax3.set_ylim([-1, 1])
    ax3.set_aspect(1.0/ax3.get_data_ratio(), adjustable='box')
    plt.title('(d)$\\beta=20$',y=-0.2)
    ax3.set_xlabel('$t$',loc='left')
    ax3.set_ylabel('$x$',loc='bottom')
   
    ax4=plt.subplot(2,3,5)
    ps,uniform500=xiaobo.fusample(model,step=30)
    ax4.scatter(uniform500[:,0],uniform500[:,1],s=1,c='salmon')
    ax4.set_xlim([0, 0.6])  
    ax4.set_ylim([-1, 1])
    ax4.set_aspect(1.0/ax4.get_data_ratio(), adjustable='box')
    plt.title('(e)$\\beta=30$',y=-0.2)
    ax4.set_xlabel('$t$',loc='left')
    ax4.set_ylabel('$x$',loc='bottom')
   
    ax5=plt.subplot(2,3,6)
    ps,uniform500=xiaobo.fusample(model,step=40)
    ax5.scatter(uniform500[:,0],uniform500[:,1],s=1,c='salmon')
    ax5.set_xlim([0, 0.6])  
    ax5.set_ylim([-1, 1])
    ax5.set_aspect(1.0/ax5.get_data_ratio(), adjustable='box')
    plt.title('(f)$\\beta=40$',y=-0.2)
    ax5.set_xlabel('$t$',loc='left')
    ax5.set_ylabel('$x$',loc='bottom')
    plt.show()


'''

    N = 202
    tspace = np.linspace(0., 1, N )
    xspace = np.linspace(-1.,1., N )
    T, X = np.meshgrid(tspace, xspace)
    Xgrid = np.vstack([T.flatten(),X.flatten()]).T
   
    fig = plt.figure(figsize=(6,6))
    plt.scatter(Xgrid[:,0],Xgrid[:,1] , marker='.', alpha=0.5)
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('$(x_i,t_i)$ of size N×N')
    plt.show()
   
    upred = model(torch.tensor(Xgrid,dtype=torch.float32).cuda())
    U = upred.cpu().detach().numpy().reshape(N,N)
    img = U
    #小波变换
    import matplotlib.ticker as ticker
    plt.imshow(img)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=10)
    cbar.locator  = ticker.MaxNLocator(nbins=5)
    plt.title('$y_i$ of size N×N')
    plt.show()
    coeffs = pywt.dwt2(img, 'db1')
    p,(q,r,s)=coeffs
    cA, (cH, cV, cD) = coeffs
    filtered_coeffs_h=(np.zeros_like(p),(q,r,s))
    filtered_img_h = pywt.idwt2(filtered_coeffs_h, 'db1') 
    #filtered_img_h = cv2.normalize(filtered_img_h, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #filtered_img_h = cv2.equalizeHist(filtered_img_h) 
    plt.imshow(filtered_img_h)

    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=10)
    cbar.locator  = ticker.MaxNLocator(nbins=5)
    plt.title('$y_i^\'$ of size N×N')
    plt.show()
   
    ax1=plt.subplot(2,2,1)
    plt.imshow(cA)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=10)
    cbar.locator  = ticker.MaxNLocator(nbins=5)
    plt.title('LL')
    
    plt.subplot(2,2,2)
    plt.imshow(cH)
    cbar=plt.colorbar(format='%.0e')
    cbar.ax.tick_params(labelsize=10)
    cbar.locator  = ticker.MaxNLocator(nbins=5)
    plt.title('HL')

    plt.subplot(2,2,3)
    plt.imshow(cV)
    cbar=plt.colorbar(format='%.0e')
    cbar.ax.tick_params(labelsize=10)
    cbar.locator  = ticker.MaxNLocator(nbins=5)
    plt.title('LH')

    plt.subplot(2,2,4)
    plt.imshow(cD)
    cbar=plt.colorbar(format='%.0e')
    cbar.ax.tick_params(labelsize=10)
    cbar.locator  = ticker.MaxNLocator(nbins=5)
    plt.title('HH')
    plt.show()
    

    # 将各个子图进行拼接，最后得到一张图
    cA=np.zeros_like(cA)
    AH = np.concatenate([cA, cH], axis=1)
    VD = np.concatenate([cV, cD], axis=1)
    img = np.concatenate([AH, VD], axis=0)
    #img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #img = cv2.equalizeHist(img)  
    plt.imshow(img)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=10)
    cbar.locator  = ticker.MaxNLocator(nbins=5)
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D
    N = 1000
    tspace = np.linspace(0., 1., N + 1)
    xspace = np.linspace(-1.,1., N + 1)
    T, X = np.meshgrid(tspace, xspace)
    Xgrid = np.vstack([T.flatten(),X.flatten()]).T
    upred = model.cpu()(torch.tensor(Xgrid,dtype=torch.float32))
    U = upred.detach().numpy().reshape(N+1,N+1)

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, U, cmap='viridis');
    ax.view_init(35,35)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$u_\\theta(t,x)$')
    ax.set_title("PINN solution for Burgers equation");
   # plt.savefig('My_Solution.png', bbox_inches='tight', dpi=300);
    plt.show()
'''
evaluation_setup()
