import os
import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle

white = (255,255,255)
black = (0, 0, 0)
red = (255,0,0)
green = (0,255,0)
blue = (0, 255, 255)
gray = (165, 165, 165)

IM_DIM = 512
CHANNELS = 3
bar_width = IM_DIM//32

####################
##### HELPERS ######
####################

def four_plus_poly_pts(xmin, xmax, ymin, ymax, sides, offset):
    '''
    Generate polygons with sides > 4 using points from unit circle
    '''
    r = (xmax - xmin)//2 + offset + 10
    x0, y0 = xmin + int(r*1.5) - offset, ymin + int(r*1.5) - offset

    shape_coords = [[x0, y0]]

    x_update = lambda x0, i, n: int(x0 - r * np.cos(2*np.pi*(i/n)))
    y_update = lambda y0, i, n: int(y0 - r * np.sin(2*np.pi*(i/n)))
    # iterate around unit circle
    for i in range(sides):
        x0, y0 = x_update(x0, i, 5), y_update(y0, i, 5)

        shape_coords.append([x0, y0])

    shape_coords = np.array([shape_coords])
    
    return shape_coords

# Placement of vertices for objects
def get_shape_coords(shape, orientation, offset=0):
    '''
    Parameters:
        shape (str): Desired shape
        orientation (str): Orientation of the shapes in the image
        offset (int): Used to create larger or smaller versions of shapes
    '''
    orient_ops = ['horizontal', 'vertical', 'diagonal']
    shape_ops = ['square', 'triangle', 'circle', 'pentagon', 'hexagon']
    
    assert orientation in orient_ops, f"Please select from the following orientations: {', '.join(orient_ops)}"
    assert shape in shape_ops, f"Please select from the following shapes: {', '.join(shape_ops)}"
    
    
    shape1_coords = None
    shape2_coords = None
    # shape area hyperparameters
    if orientation == 'horizontal':
        xmin1, xmax1, ymin1, ymax1 = 60, 160, 200, 300
        xmin2, xmax2, ymin2, ymax2 = 355, 455, 200, 300
    elif orientation == 'vertical':
        xmin1, xmax1, ymin1, ymax1 = 200, 300, 50, 150
        xmin2, xmax2, ymin2, ymax2 = 200, 300, 350, 450
    elif orientation == 'diagonal':
        xmin1, xmax1, ymin1, ymax1 = 50, 150, 50, 150
        xmin2, xmax2, ymin2, ymax2 = 350, 450, 350, 450
    
        
    if shape == 'square':
        shape1_coords = np.array([[[xmin1-offset, ymin1-offset], [xmin1-offset, ymax1+offset],
                                   [xmax1+offset, ymax1+offset], [xmax1+offset, ymin1-offset]]], dtype=np.int32)
        shape2_coords = np.array([[[xmin2-offset, ymin2-offset], [xmin2-offset, ymax2+offset],
                                   [xmax2+offset, ymax2+offset], [xmax2+offset, ymin2-offset]]], dtype=np.int32)
    elif shape == 'triangle':
        shape1_coords = np.array([[[xmin1 + (xmax1-xmin1)//2, ymin1-offset], 
                                   [xmin1-offset, ymax1], [xmax1+offset, ymax1]]], dtype=np.int32)
        shape2_coords = np.array([[[xmin2 + (xmax2-xmin2)//2, ymin2-offset], 
                                   [xmin2-offset, ymax2], [xmax2+offset, ymax2]]], dtype=np.int32)
    elif shape == 'circle':
        shape1_coords = (xmin1+(xmax1-xmin1)//2, ymin1+(ymax1-ymin1)//2)
        shape2_coords = (xmin2+(xmax2-xmin2)//2, ymin2+(ymax2-ymin2)//2)
    elif shape == 'pentagon':
        shape1_coords = four_plus_poly_pts(xmin1, xmax1, ymin1, ymax1, 5, offset)
        shape2_coords = four_plus_poly_pts(xmin2, xmax2, ymin2, ymax2, 5, offset)
            
    return shape1_coords, shape2_coords


# Draw occluding stripes on image
def draw_stripes(base, bar_width, im_dim, base_color, 
                 stripe_color, stripe_or, shape_or, 
                 illusion=False):
    assert stripe_or in ['vertical', 'horizontal', 'diagonal'], 'Enter valid stripe orientation'
    assert shape_or in ['vertical', 'horizontal', 'diagonal'], 'Enter valid shape orientation'
    
    
    # Usual case: when stripe orientation is different than object orientation stripes span whole image for the illusion
    #     In unusual case the stripes must only span half the image to generate the illusion
    if stripe_or != shape_or or not illusion:
        if stripe_or == 'vertical':
            start_x = 0
            while start_x < im_dim: 

                # reset start x at halfway point for illusion (makes sure stripes are aligned correctly)
                if illusion and start_x == im_dim//2:
                    start_x = im_dim//2 + bar_width
                    
                # coordinates for stripe
                stripe_pts = np.array([[[start_x, 0], [start_x, im_dim-1], 
                                        [start_x+bar_width-1, im_dim-1], [start_x+bar_width-1, 0]]])
        
                # switch stripe color for illusion at halfway
                if illusion and start_x >= im_dim//2:
                    cv.fillPoly(base, stripe_pts, base_color)
                else:
                    cv.fillPoly(base, stripe_pts, stripe_color)
                    
                start_x = start_x + 2*bar_width                

        elif stripe_or == 'horizontal':
            start_y = bar_width
            while start_y < im_dim:

                if illusion and start_y == im_dim//2 + bar_width:
                    start_y = im_dim//2

                stripe_pts = np.array([[[0,start_y], [0, start_y+bar_width-1],
                                       [im_dim, start_y+bar_width-1], [im_dim, start_y]]])

                if illusion and start_y >= im_dim//2:
                    cv.fillPoly(base, stripe_pts, base_color)
                else:
                    cv.fillPoly(base, stripe_pts, stripe_color)

                start_y = start_y + 2*bar_width
                
    elif stripe_or == shape_or and stripe_or != 'diagonal' and illusion: # this case is already handled for diagonal
        if stripe_or == 'vertical':
            # top stripes (non-illusion portion)
            start_x = 0
            while start_x < im_dim: 
                # coordinates for stripe
                stripe_pts = np.array([[[start_x, 0], [start_x, im_dim//2], 
                                        [start_x+bar_width-1, im_dim//2], [start_x+bar_width-1, 0]]])
                cv.fillPoly(base, stripe_pts, stripe_color)

                start_x = start_x + 2*bar_width 
            # bottom stripes (non-illusion portion)
            start_x = bar_width
            while start_x < im_dim: 
                # coordinates for stripe
                stripe_pts = np.array([[[start_x, im_dim//2], [start_x, im_dim], 
                                        [start_x+bar_width-1, im_dim], [start_x+bar_width-1, im_dim//2]]])
                cv.fillPoly(base, stripe_pts, base_color)

                start_x = start_x + 2*bar_width  
        elif stripe_or == 'horizontal':
            # left stripes (non-illusion portion)
            start_y = bar_width
            while start_y < im_dim: 
                # coordinates for stripe
                stripe_pts = np.array([[[0, start_y], [0, start_y+bar_width-1], 
                                        [im_dim//2, start_y+bar_width-1], [im_dim//2, start_y]]])
                cv.fillPoly(base, stripe_pts, stripe_color)

                start_y = start_y + 2*bar_width 
            # right stripes (illusion portion)
            start_y = 0
            while start_y < im_dim: 
                # coordinates for stripe
                stripe_pts = np.array([[[im_dim//2, start_y], [im_dim//2, start_y+bar_width-1], 
                                        [im_dim, start_y+bar_width-1], [im_dim, start_y]]])
                cv.fillPoly(base, stripe_pts, base_color)

                start_y = start_y + 2*bar_width 
                
    # same process for illusion and non-illusions
    if stripe_or == 'diagonal': 
        xleft = bar_width
        xright = bar_width*2
        ytop = bar_width
        ybottom = bar_width*2

        while xleft < im_dim:
            stripe_pts = np.array([[[xleft,0], [0, ytop],
                                   [0, ybottom], [xright, 0]]])

            # base color 
            cv.fillPoly(base, stripe_pts, stripe_color)

            xleft = xleft + 2*bar_width
            xright = xright + 2*bar_width
            ytop = ytop + 2*bar_width
            ybottom = ybottom + 2*bar_width

        xleft = 1 if illusion else bar_width
        xright = bar_width-1 if illusion else bar_width*2
        ytop = 1 if illusion else bar_width
        ybottom = bar_width-1 if illusion else bar_width*2
        while ytop < im_dim:
            stripe_pts = np.array([[[xleft, im_dim], [xright, im_dim],
                                   [im_dim, ybottom], [im_dim, ytop]]])

            # base color 
            if illusion:
                cv.fillPoly(base, stripe_pts, base_color)
            else:
                cv.fillPoly(base, stripe_pts, stripe_color)

            xleft = xleft + 2*bar_width
            xright = xright + 2*bar_width
            ytop = ytop + 2*bar_width
            ybottom = ybottom + 2*bar_width



#######################
###### GENERATOR ######
#######################
def generate_illusion_set(
    base_color=(255,255,255),
    shape='square',
    shape_color=(165,165,165),
    shape_size='normal',
    shape_or='horizontal',
    stripe_or='vertical'
):
    '''
    Function to generate a non-illusion image and an illusion counterpart,
    controlling for other qualia. The base image is also provided for reference.
    Illusions are generated by projecting two identical objects onto the base
    image and then occluding both objects by white or black stripes. The illusion 
    is generated by occluding each shape with different colored stripes.
    
    Parameters:
        base_color (tuple): background color
        shape (str): subject of illusion
        shape_color (tuple): RGB
        shape_size (str): one of three shape settings (small/normal/large)
        shape_or (str): orientation of shapes w.r.t each other (horizontal/vertical/diagonal)
        stripe_or (str): orientation of the occluding stripes (horizontal/vertical/diagonal)
        
    Return:
        A tuple of three, 3x512x512 numpy arrays. (1) Base image, (2) Non-illusion, (3) Illusion 
    '''
    
    ## SET UP IMAGE ##
    IM_DIM = 512
    CHANNELS = 3
    BAR_WIDTH = IM_DIM//128 # at 128 the images start and end with the same color stripe

    BASE_RADIUS = 64
    
    white = (255,255,255)
    black = (0, 0, 0)
    
    if shape_size == 'small':
        shape_scale = 0.75
        offset = -10
    elif shape_size == 'normal':
        shape_scale = 1
        offset = 0
    elif shape_size == 'large':
        shape_scale = 1.5
        offset = 15
    
    # set default base color
    if np.sum(base_color) == 0:
        base = np.zeros((IM_DIM,IM_DIM,CHANNELS), dtype=np.uint8)
        oppos_color = white
    elif np.sum(base_color) == int(255*3):
        base = 255*np.ones((IM_DIM,IM_DIM,CHANNELS), dtype=np.uint8)
        oppos_color = black
    else:
        print('Invalid base, must be white or black')
        return

    base = base
    nonillu_base = base.copy()
    illu_base = base.copy()
    
    # shape positions
    shape1_pts, shape2_pts = get_shape_coords(shape, shape_or, offset)

    ## DRAW IMAGES ##
    if shape == 'circle':
        ## BASE  ##
        cv.circle(base, shape1_pts, int(shape_scale*50), shape_color, thickness=-1)
        cv.circle(base, shape2_pts, int(shape_scale*50), shape_color, thickness=-1)
        
        ## NON-ILLUSION ##
        cv.circle(nonillu_base, shape1_pts, int(shape_scale*50), shape_color, thickness=-1)
        cv.circle(nonillu_base, shape2_pts, int(shape_scale*50), shape_color, thickness=-1)
        
    else:
        ## BASE  ##
        cv.fillPoly(base, shape1_pts, shape_color)
        cv.fillPoly(base, shape2_pts, shape_color)

        ## NON-ILLUSION ##
        cv.fillPoly(nonillu_base, shape1_pts, shape_color)
        cv.fillPoly(nonillu_base, shape2_pts, shape_color)
    
    # stripes for non-illusion
    draw_stripes(nonillu_base, BAR_WIDTH, IM_DIM, base_color, oppos_color, 
                 stripe_or=stripe_or, shape_or=shape_or, illusion=False) 
    
    # ILLUSION ##
    
    # Apply illusion overlay (right/bottom/bottom right side default for now)
    if stripe_or != shape_or:
        if stripe_or == 'vertical':
            illu_overlay = np.array([[[IM_DIM//2, 0], [IM_DIM//2, IM_DIM], [IM_DIM, IM_DIM], [IM_DIM, 0]]])
        elif stripe_or == 'horizontal':
            illu_overlay = np.array([[[0, IM_DIM//2], [0, IM_DIM], [IM_DIM, IM_DIM], [IM_DIM, IM_DIM//2]]])
    elif stripe_or != 'diagonal':
        if stripe_or == 'vertical':
            illu_overlay = np.array([[[0, IM_DIM//2], [0, IM_DIM], [IM_DIM, IM_DIM], [IM_DIM, IM_DIM//2]]])
        elif stripe_or == 'horizontal':
            illu_overlay = np.array([[[IM_DIM//2, 0], [IM_DIM//2, IM_DIM], [IM_DIM, IM_DIM], [IM_DIM, 0]]])
    if stripe_or == 'diagonal':
        illu_overlay = np.array([[[IM_DIM+1,0], [0, IM_DIM+1], [IM_DIM, IM_DIM]]])
    cv.fillPoly(illu_base, illu_overlay, oppos_color)
    # apply shapes
    if shape == 'circle':
        cv.circle(illu_base, shape1_pts, int(shape_scale*50), shape_color, thickness=-1)
        cv.circle(illu_base, shape2_pts, int(shape_scale*50), shape_color, thickness=-1)
    else:
        cv.fillPoly(illu_base, shape1_pts, shape_color)
        cv.fillPoly(illu_base, shape2_pts, shape_color)
    
    # Stripes for illusion
    draw_stripes(illu_base, BAR_WIDTH, IM_DIM, base_color, oppos_color, 
                 stripe_or=stripe_or, shape_or=shape_or, illusion=True)

    return base, nonillu_base, illu_base