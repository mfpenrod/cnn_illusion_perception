import sys
sys.path.append('./')
from data_gen import generate_illusion_set

import os
import pickle


white = (255,255,255)
black = (0, 0, 0)
red = (255,0,0)
green = (0,255,0)
blue = (0, 255, 255)
gray = (165, 165, 165)
def save_dataset(data_path):
    bases_path = data_path + '/bases'
    non_illus_path = data_path + '/non_illu'
    illus_path = data_path + '/illu'

    paths = [bases_path, non_illus_path, illus_path]
    headers = ['base', 'nonillu', 'illu']
    for path in paths:
        if not os.path.isdir(path):
            os.mkdir(path)

    base_colors = [white, black] # determines whether illusion makes object lighter or darker, respectively
    shapes = ['circle', 'square', 'triangle', 'pentagon']
    colors = [blue, red, green, gray]
    shape_sizes = ['small', 'normal', 'large']
    shape_ors = ['vertical', 'horizontal', 'diagonal']
    stripe_ors = ['vertical', 'horizontal', 'diagonal']
    n_image_sets = len(base_colors) * len(shapes) * len(colors) * len(shape_sizes) * len(shape_ors) * len(stripe_ors)

    im_counter = -1
    for base_color in base_colors:
        for shape in shapes:
            for color in colors:
                for shape_size in shape_sizes:
                    for shape_or in shape_ors:
                        for stripe_or in stripe_ors:
                            im_counter += 1
                            
                            #get images
                            ims = generate_illusion_set(
                                    base_color=base_color,
                                    shape=shape,
                                    shape_color=color,
                                    shape_size=shape_size,
                                    shape_or=shape_or,
                                    stripe_or=stripe_or
                                    )
                            # save
                            for i, im in enumerate(ims):
                                        
                                file_name = f'{paths[i]}/{headers[i]}_{im_counter}_{shape}.pkl'
                                with open(file_name, 'wb') as f:
                                    pickle.dump(im, f)
                        
                                if i in [0, 1]: # base or non-illusion
                                    train_test_path = f'{data_path}/train/{headers[i]}_{im_counter}_{shape}.pkl'
                                elif i==2:
                                    train_test_path = f'{data_path}/test/{headers[i]}_{im_counter}_{shape}.pkl'
                                    
                                if not os.path.isdir('/'.join(train_test_path.split('/')[:-1])):
                                    os.mkdir('/'.join(train_test_path.split('/')[:-1]))
                                with open(train_test_path, 'wb') as f:
                                    pickle.dump(im, f)
                            
                            if im_counter%49 == 0:
                                print(f'Saved {im_counter+1}/{n_image_sets} image sets')
        
