import numpy as np

def normalize(x):
    '''
    Normalization of coordinates (centroid to the origin and: number of dimensions, 3 here
    x: the data to be normalized (directions at different columns and points at rows)
    Output
    ------
    Tr: the transformation matrix (translation plus scaling)
    x: the transformed data
    '''

    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    
    Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    Tr = np.linalg.inv(Tr)
    
    x = np.dot( Tr, np.concatenate( (x.T, np.ones((1,x.shape[0]))) ) )
    x = x[0:2, :].T

    return x, Tr

def dlt(xy, uv, normalization=False):
    '''
    Camera calibration by DLT using known flat object points and their image points.
    Input
    -----
    xy: coordinates in the object 2D space.
    uv: coordinates in the image 2D space.
    The coordinates (x,y and u,v) are given as columns and the different points as rows.
    There must be at least 4 calibration points for the 2D DLT.
    Output
    ------
     H: matrix of 8 parameters of the calibration matrix.
     err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
    '''
    
    # Converting all variables to numpy array
    xy = np.asarray(xy)
    uv = np.asarray(uv)

    n = xy.shape[0]

    # Validating the parameters:
    if uv.shape[0] != n:
        raise ValueError('Object ({} points) and image ({} points) have different number of points.'.format(n, uv.shape[0]) )

    if (xy.shape[1] != 2):
        raise ValueError('Incorrect number of object coordinates ({} points) for DLT (it should be {}).'.format(xy.shape[1], 2) )

    if (uv.shape[1] != 2):
        raise ValueError('Incorrect number of image coordinates ({} points) for DLT (it should be {}).'.format(uv.shape[1], 2) )

    if (n < 4):
        raise ValueError('2D DLT requires at least {} calibration points. Only {} points were entered.'.format(4, n) )
        
    # Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
    # This is relevant when there is a considerable perspective distortion.
    # Normalization: mean position at origin and mean distance equals to 1 at each direction.

    if normalization:
        xyn, Txy = normalize(xy)
        uvn, Tuv = normalize(uv)
    else:
        xyn, Txy = xy, np.diag([1, 1, 1]) 
        uvn, Tuv = uv, np.diag([1, 1, 1])

    A = []

    for i in range(n):
        x, y = xyn[i, 0], xyn[i, 1]
        u, v = uvn[i, 0], uvn[i, 1]
        A.append( [x, y, 1, 0, 0, 0, -u * x, -u * y, -u] )
        A.append( [0, 0, 0, x, y, 1, -v * x, -v * y, -v] )

    # Convert A to array
    A = np.asarray(A) 
    #print("A"); print(A)

    # Find the parameters (8 + 1 multiplier):
    u, s, vh = np.linalg.svd(A)

    # The parameters are in the last line of vh and normalize them
    L = vh[-1, :] / vh[-1, -1]
    # Camera projection matrix
    H = L.reshape(3, 3)
    #print("H"); print(H)
    
    if normalization:
        # Denormalization
        # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
        H = np.dot( np.dot( np.linalg.pinv(Tuv), H ), Txy )
        H = H / H[-1, -1]

    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    uv2 = np.dot( H, np.concatenate( (xy.T, np.ones((1, xy.shape[0]))) ) ) 
    uv2 = uv2 / uv2[2, :] 
    # Mean distance:
    err = np.sqrt( np.mean(np.sum( (uv2[0:2, :].T - uv)**2, 1)) ) 

    return H.T, err

def transform(xy, T):
    xyh = np.concatenate((xy,  np.ones((xy.shape[0], 1)) ), axis=1)
    uvh = np.dot(xyh, T)
    uvh = uvh / uvh[:, 2][:, None]
    return uvh[:, 0:2]

if __name__ == "__main__":
    # Known 3D coordinates
    xy = np.array([
        [0, 2],
        [1, 2],
        [2, 2],
        [2, 1],
        [2, 0],
        [1, 0],
        [0, 0],
        [0, 1],
        [1, 1]
    ])
    print("Object coordinates"); print(xy)

    # Known pixel coordinates
    T = np.array([
        [130, 50, 0.08],
        [-40, 90, 0.3],
        [80, 50, 1]
        ])
    print("Real transform matrix"); print(T)
    uv = transform(xy, T)
    print("Image coordinates"); print(uv)

    uv_noisy = uv + np.random.normal(0, 5, uv.shape)

    P, err = dlt(xy, uv_noisy, normalization=True)
    print("Transform matrix restored from noisy data"); print(P)
    print("Error"); print(err)

    import matplotlib.pyplot as plt
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(xy[:,0], xy[:,1], 'b', label="Исходные точки на дороге")
    ax1.plot(xy[:,0], xy[:,1], 'b.')
    ax1.grid() 
    ax1.set_aspect('equal')

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(uv[:,0], uv[:,1], 'b', label="Исходные точки на изображении")
    ax2.plot(uv[:,0], uv[:,1], 'b.')
    ax2.plot(uv_noisy[:,0], uv_noisy[:,1], 'g--', label="Исходные точки на изображении с добавленным шумом")
    ax2.plot(uv_noisy[:,0], uv_noisy[:,1], 'g.')
    ax2.grid() 
    ax2.set_aspect('equal')

    xy = np.array([ [1+np.cos(t), 1+np.sin(t)] for t in np.linspace(0, 2*np.pi, 100) ])
    uv = transform(xy, T)

    ax1.plot(xy[:,0], xy[:,1], 'r--', label="Истинное расположение второй фигуры на дороге")
    
    P_inv = np.linalg.inv(P)
    P_inv = P_inv / P_inv[-1, -1]
    xy_restored = transform(uv, P_inv)

    ax1.plot(xy_restored[:,0], xy_restored[:,1], 'r', label="Восстановленное расположение второй фигуры на дороге")
    ax2.plot(uv[:,0], uv[:,1], 'r', label="Вторая фигура на изображеннии")

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, fontsize=5)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, fontsize=5)

    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    #plt.show()

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages('test.pdf') as pdf:
        pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
        pass
