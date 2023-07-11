import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

seed = 42

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import torch
torch.use_deterministic_algorithms(True)
torch.manual_seed(seed)

import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse() # creation of a BaseOptions objet with all the arguments of BaseOptions+TrainOptions. Run parse fct 
data_loader = CreateDataLoader(opt) #create a CustomDatasetDataLoader object with all paths and transformations
dataset = data_loader.load_data()
dataset_size = len(data_loader) # get the number of images in the dataset.
print('#training images = %d' % dataset_size)
model = create_model(opt) # create a model given opt.model and other options
visualizer = Visualizer(opt) # create a visualizer that display/save images and plots

CRITIC_ITERS = 5
total_steps = 0
iter_d = 0
only_d = False
step_opti_G = 3

for epoch in range(1, opt.niter + opt.niter_decay + 1): # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time() # timer for entire epoch
    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time() # timer for computation per iteration
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        # y,x,w,h = data['bbox']
        # if w[0]-x[0] < 39:
        #     continue
        model.set_input(data)
#         if iter_d <= CRITIC_ITERS-1:
#             only_d = False
#         else:
#             only_d = False
        model.optimize_parameters_D() # calculate loss functions, get gradients, update network weights
        if only_d == False:
            for i in range (step_opti_G) :
                model.optimize_parameters_G()  

        if total_steps % opt.display_freq == 0: # display images on visdom
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:  # print training losses and save logging information to the disk
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0: # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
        iter_d += 1
        if iter_d == 6:
            iter_d = 0

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
