from openpiv import tools, pyprocess, validation, filters, scaling, preprocess
import numpy as np

import matplotlib.pyplot as plt

import imageio
#完成了多张图片的piv流场速度提取和云图以及涡量云图，调参可以改变图像细节丰富程度，但会使效果变差
#显示的图片依次为u,v,w的云图以及速度矢量图
def process_base(file_a,file_b,counter):

    #file_a='60ms00000000'+'1'+'.bmp'
    #file_b='60ms000000002.bmp' 
    frame_a  = tools.imread( file_a )  #从指定的文件读取图像,返回一个带有灰度级的NUMPY数组
    frame_b  = tools.imread( file_b )
    #fig,ax = plt.subplots(1,2,figsize=(12,10))
    
    winsize = 24 #调参调这里 查询窗口的大小
    searchsize = 32 #调参调这里 第二帧图像的搜索范围
    overlap = 12 #调参调这里 两个相邻窗口重叠的像素数
    dt = 8/258 #调参调这里 两帧图像的时间间隔

    #遮蔽物体
    #frame_a, _ = preprocess.dynamic_masking(frame_a,method='intensity',filter_size=7,threshold=0.01)
    #frame_b, _ = preprocess.dynamic_masking(frame_b,method='intensity',filter_size=7,threshold=0.01)
    #plt.imshow(np.c_[frame_a,frame_b],cmap='gray')

    u0, v0, sig2noise = pyprocess.extended_search_area_piv(frame_a.astype(np.int32), 
                                                           frame_b.astype(np.int32), 
                                                           window_size=winsize, 
                                                           overlap=overlap, 
                                                           dt=dt, 
                                                           search_area_size=searchsize, 
                                                           sig2noise_method='peak2peak')
    #输出速度数组（单位：像素/秒）和对应的信噪比

    #查询窗口的中心坐标
    x, y = pyprocess.get_coordinates( image_size=frame_a.shape, 
                                     search_area_size=searchsize, 
                                     overlap=overlap )
    
    '''
    Validation
    '''
    #0.整体大小过滤，将给定范围外的速度替换
    u1, v1, mask0 = validation.global_val( u0, v0, (-480.,480.),(-480,480.))

    # 绘制信噪比分布直方图
    #plt.hist(sig2noise.flatten())
    p = np.percentile(sig2noise,10) # bottom %
    #plt.plot([p,p],[0,100],lw=2)
    #寻找合适的信噪比阈值
    
    #1.信噪比过滤，从互相关信号与噪声比中消除错误矢量,如果信噪比低于指定阈值，则将错误矢量替换为零，mask标记是否替换
    u1, v1, mask1 = validation.sig2noise_val( u1, v1, sig2noise, threshold = p )

    #2.局部中值过滤
    u1, v1, mask2 = validation.local_median_val(u1, v1, u_threshold=100, v_threshold=100, size=1)

    #合并各种过滤方法的mask
    mask = mask0 | mask1 | mask2                                        
    
    #使用迭代图像修复算法替换速度场中的无效矢量
    u2, v2 = filters.replace_outliers( u1, v1, 
                                      method='localmean', 
                                      max_iter=10, #迭代次数
                                      kernel_size=2)

    #均匀缩放
    x, y, u3, v3 = scaling.uniform(x, y, u2, v2, 
                                   scaling_factor = 96.52 # 图像缩放比例，以每米像素为单位
                                   )

    #图像坐标系和物理坐标系相互转换。图像坐标系：原点为左上顶点，x向右，y向下。物理坐标系：原点为左下，右手系
    x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

    file_data='epx_'+str(counter).zfill(3)+'.txt'
    tools.save(x, y, u3, v3, mask, file_data )  #Save flow field to an ascii file

    '''
    cset = plt.contourf(x,y,u3)
    contour = plt.contour(x,y,u3)
    #plt.clabel(contour,fontsize=10,colors='k')
    plt.colorbar(cset)
    plt.show()


    cset = plt.contourf(x,y,v3)
    contour = plt.contour(x,y,v3)
    #plt.clabel(contour,fontsize=10,colors='k')
    plt.colorbar(cset)
    plt.show()

    x_num1 = len(x)
    x_num2 = len(x[0])
    #print(x_num2)
    vorticity = np.empty([x_num1-2,x_num2-2],dtype=float)
    x_w = np.empty([x_num1-2,x_num2-2],dtype=float)
    y_w = np.empty([x_num1-2,x_num2-2],dtype=float)

    #计算涡量
    for i in range(1,x_num1-1):
        for j in range(1,x_num2-1):
            vorticity[i-1][j-1] = (u3[i+1][j]-u3[i-1][j])/(y[i+1][j]-y[i-1][j])-(v3[i][j+1]-v3[i][j-1])/(x[i][j+1]-x[i][j-1])  #∂u/∂y-∂v/∂x
            x_w[i-1][j-1] = x[i][j]
            y_w[i-1][j-1] = y[i][j]

    cset = plt.contourf(x_w,y_w,vorticity)
    contour = plt.contour(x_w,y_w,vorticity)
    #plt.clabel(contour,fontsize=10,colors='k')
    plt.colorbar(cset)
    plt.show()
    '''

    #速度场矢量图
    fig, ax = plt.subplots(figsize=(8,8))
    tools.display_vector_field(file_data, 
                               ax=ax, scaling_factor=96.52, 
                               scale=50, 
                               width=0.0035, 
                               on_img=True, 
                               image_name=file_a)

for i in range(2):#在这里改循环次数，也就是  （bmp的数量-1）

    file_a = 'z'+str(i+1).zfill(3)+'.bmp' #测试时将bmp文件放在同一目录中，如在其他文件夹中或文件名不一样，直接修改file_a和file_b
    file_b = 'z'+str(i+2).zfill(3)+'.bmp' #左边补0到三位数字
    process_base(file_a,file_b,i+1)