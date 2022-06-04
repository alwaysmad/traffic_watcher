import numpy as np
import cv2 as cv
from scipy.optimize import linprog
from itertools import pairwise

from dlt import transform

def draw_transformed_line(image, xy1, xy2, T, T_inv, colorBGR ):
    W = image.shape[1]
    H = image.shape[0]

    xyh1 = np.array([
        [xy1[0], xy1[1], 1]
        ])

    xyh2 = np.array([
        [xy2[0], xy2[1], 1]
        ])

    uvh1 = np.dot(xyh1, T)
    uvh2 = np.dot(xyh2, T)

    u1 = uvh1[0, 0]; v1 = uvh1[0, 1]; h1 = uvh1[0, 2]
    u2 = uvh2[0, 0]; v2 = uvh2[0, 1]; h2 = uvh2[0, 2]

    A =  (v1*h2 - v2*h1)
    B = -(u1*h2 - u2*h1)
    C =  (u1*v2 - u2*v1)

    t13 = T_inv[0, 2]; t23 = T_inv[1, 2]; t33 = T_inv[2, 2]
    
    A_eq = [
            [A, B]
            ]
    b_eq = [-C]

    A_ub = [
            [-t13, -t23]
            ]
    b_ub = [-0 + t33]
    #b_ub = [-0.1 + t33]

    uv_1 = linprog( c=[A, -B], A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=[(0,W), (0,H)])
    #print(uv_1)
    uv_1 = uv_1.x[0:2]
    uv_2 = linprog( c=[-A, B], A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=[(0,W), (0,H)])
    #print(uv_2); 
    uv_2 = uv_2.x[0:2]

    cv.line(image, 
        (int(uv_1[0]), int(uv_1[1])),
        (int(uv_2[0]), int(uv_2[1])),
        colorBGR, 1, cv.LINE_AA)

from math import ceil, floor

def draw_transformed_grid(image, T, T_inv, step=None):
    W = image.shape[1]
    H = image.shape[0]

    t13 = T_inv[0, 2]; t23 = T_inv[1, 2]; t33 = T_inv[2, 2]

    A_ub = [
            [-t13, -t23]
            ]
    b_ub = [-0.30 + t33]

    uv_1 = linprog(c = [1, 1], A_ub=A_ub, b_ub=b_ub, bounds=[(0,W), (0,H)]) # u min
    uv_1 = uv_1.x
    uv_2 = linprog(c = [-1, 1], A_ub=A_ub, b_ub=b_ub, bounds=[(0,W), (0,H)]) # u max
    uv_2 = uv_2.x
    uv_3 = linprog(c = [1, -1], A_ub=A_ub, b_ub=b_ub, bounds=[(0,W), (0,H)]) # v min
    uv_3 = uv_3.x
    uv_4 = linprog(c = [-1, -1], A_ub=A_ub, b_ub=b_ub, bounds=[(0,W), (0,H)]) # v max
    uv_4 = uv_4.x

    xy_1 = transform(np.array([[uv_1[0], uv_1[1]]]), T_inv)
    xy_2 = transform(np.array([[uv_2[0], uv_2[1]]]), T_inv)
    xy_3 = transform(np.array([[uv_3[0], uv_3[1]]]), T_inv)
    xy_4 = transform(np.array([[uv_4[0], uv_4[1]]]), T_inv)

    x_l = min( xy_1[0,0], xy_2[0,0], xy_3[0,0], xy_4[0,0] )
    x_u = max( xy_1[0,0], xy_2[0,0], xy_3[0,0], xy_4[0,0] )
    y_l = min( xy_1[0,1], xy_2[0,1], xy_3[0,1], xy_4[0,1] )
    y_u = max( xy_1[0,1], xy_2[0,1], xy_3[0,1], xy_4[0,1] )
    
    if step is None:
        d =  min(x_u-x_l, y_u-y_l)
        if d <= 200:
            step = 1
        elif d <= 2000:
            step = 10
        elif d <= 20000:
            step = 100
        else:
            step = 1000

    for x in range( floor(x_l), ceil(x_u), step ):
        draw_transformed_line(image, (x,0), (x,1), T, T_inv, (0,0,255))
    for y in range( floor(y_l), ceil(y_u), step ):
        draw_transformed_line(image, (0,y), (1,y), T, T_inv, (255,0,0))

####################################################################################################

if __name__ == "__main__":
    
    points_of_square = [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 0]
        ]
    T = np.array([
        [ 1.71318226e+02,  3.11112701e+01,  1.00227082e-01],
        [-6.44909167e+02, -3.28464273e+02, -7.30021464e-01],
        [ 8.86000000e+02,  5.08000000e+02,  1.00000000e+00],
        ])
    T_inv = np.linalg.inv(T) 

    image = np.zeros((2000,2000,3), np.uint8)

    zero_point = transform(np.array([[0, 0]]), T)
    cv.circle(image, (int(zero_point[0,0]), int(zero_point[0,1])), 4, (255,255,255), -1, cv.LINE_AA)

    draw_transformed_grid(image, T, T_inv)

    #draw_transformed_line(image, (0,0), (1,1), T, T_inv, (255,255,255) )

    # make window
    cv.namedWindow("TeSt", cv.WINDOW_NORMAL)
    cv.resizeWindow("TeSt", 800, 600) # set window size

    while True:
        cv.imshow("TeSt", image)
        if cv.waitKey(0) & 0xFF == 27: break # until ESC is pressed

