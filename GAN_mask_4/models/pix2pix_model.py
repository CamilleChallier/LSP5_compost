import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from copy import deepcopy
from . import networks
from PIL import Image
import torchvision.transforms as transforms

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # self.opt = opt
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(((0.5, ) * opt.input_nc),
                                               ((0.5, ) * opt.input_nc))]

        self.transform = transforms.Compose(transform_list)
                       
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_image = networks.define_image_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            use_sigmoid = not opt.no_lsgan
            self.netD_person = networks.define_person_D(opt.input_nc, opt.ndf, opt, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD_image, 'D_image', opt.which_epoch)
                self.load_network(self.netD_person, 'D_person', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            self.criterionGAN_image = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionGAN_person = networks.GANLoss(use_lsgan=opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_image = torch.optim.Adam(self.netD_image.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_person = torch.optim.Adam(self.netD_person.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD_image)
            networks.print_network(self.netD_person)
        print('-----------------------------------------------')

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.bbox = input['bbox']
        
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

        self.image_paths = input['A_paths' ]#if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)

        y,x,w,h = self.bbox
        self.person_crop_real = self.real_B[:,:,y[0]:h[0],x[0]:w[0]]
        self.person_crop_fake = self.fake_B[:,:,y[0]:h[0],x[0]:w[0]]

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

        y,x,w,h = self.bbox
        self.person_crop_real = self.real_B[:,:,y[0]:h[0],x[0]:w[0]]
        self.person_crop_fake = self.fake_B[:,:,y[0]:h[0],x[0]:w[0]]

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_image(self):
        """Calculate GAN loss for the discriminator"""
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD_image.forward(fake_AB.detach())
        self.loss_D_image_fake = self.criterionGAN_image(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD_image.forward(real_AB)
        self.loss_D_image_real = self.criterionGAN_image(self.pred_real, True)

        # Combined loss
        self.loss_D_image = (self.loss_D_image_fake + self.loss_D_image_real) * 0.5

        self.loss_D_image.backward()

    def backward_D_person(self):
        #Fake
        self.person_fake = self.netD_person.forward(self.person_crop_fake)
        self.loss_D_person_fake = self.criterionGAN_person(self.person_fake, False)

        #Real
        self.person_real = self.netD_person.forward(self.person_crop_real)
        self.loss_D_person_real = self.criterionGAN_person(self.person_real, True)

        #Combine loss
        self.loss_D_person = (self.loss_D_person_fake + self.loss_D_person_real) * 0.5
        self.loss_D_person.backward()


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator1 and discriminator1
        # discriminator1
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake_image = self.netD_image.forward(fake_AB)
        self.loss_G_GAN_image = self.criterionGAN_image(pred_fake_image, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        pred_fake_person = self.netD_person.forward(self.person_crop_fake)
        self.loss_G_GAN_person = self.criterionGAN_person(pred_fake_person, True)


        self.loss_G = self.loss_G_GAN_image + self.loss_G_L1 + self.loss_G_GAN_person

        self.loss_G.backward()

    def optimize_parameters_D(self):

        self.forward()
        self.optimizer_D_image.zero_grad()
        self.backward_D_image()
        self.optimizer_D_image.step()
        
        self.forward()
        self.optimizer_D_person.zero_grad()
        self.backward_D_person()
        self.optimizer_D_person.step()

    def optimize_parameters_G(self):
        
            self.forward()
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN_image', self.loss_G_GAN_image.data),
                            ('G_GAN_person', self.loss_G_GAN_person.data),
                            ('G_L1', self.loss_G_L1.data),
                            #('G_L1_person', self.loss_G_L1_person.data[0]),
                            ('D_image_real', self.loss_D_image_real.data),
                            ('D_image_fake', self.loss_D_image_fake.data),
                            ('D_person_real', self.loss_D_person_real.data),
                            ('D_person_fake', self.loss_D_person_fake.data)
                            ])

    def get_current_visuals(self, mask, size):

        if mask :
            real_A = util.tensor2im(self.real_A.data)[:,:,0:3]
            real_A_mask = util.tensor2im(self.real_A.data)[:,:,3:].reshape(size,size)
            fake_B = util.tensor2im(self.fake_B.data)[:,:,0:3]
            fake_B_mask = util.tensor2im(self.fake_B.data)[:,:,3:].reshape(size,size)
            real_B = util.tensor2im(self.real_B.data)[:,:,0:3]
            real_B_mask = util.tensor2im(self.real_B.data)[:,:,3:].reshape(size,size)
            D2_fake = util.tensor2im(self.person_crop_fake.data)[:,:,0:3]
            size_x=D2_fake.shape[0]
            size_y=D2_fake.shape[1]
            D2_real = util.tensor2im(self.person_crop_real.data)[:,:,0:3]
            D2_fake_mask = util.tensor2im(self.person_crop_fake.data)[:,:,3:].reshape(size_x,size_y)
            D2_real_mask = util.tensor2im(self.person_crop_real.data)[:,:,3:].reshape(size_x,size_y)
            y,x,w,h = self.bbox
            display = deepcopy(real_A)
            display[y[0]:h[0],x[0]:w[0],:] = D2_fake
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B), ('display', display), ('D2_fake',D2_fake),('D2_real',D2_real),('real_A_mask', real_A_mask), ('fake_B_mask', fake_B_mask), ('real_B_mask', real_B_mask), ('D2_fake_mask',D2_fake_mask),('D2_real_mask',D2_real_mask)])

        else :
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            real_B = util.tensor2im(self.real_B.data)
            D2_fake = util.tensor2im(self.person_crop_fake.data)
            D2_real = util.tensor2im(self.person_crop_real.data)
            y,x,w,h = self.bbox
            display = deepcopy(real_A)
            display[y[0]:h[0],x[0]:w[0],:] = D2_fake
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B), ('display', display), ('D2_fake',D2_fake),('D2_real',D2_real)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD_image, 'D_image', label, self.gpu_ids)
        self.save_network(self.netD_person, 'D_person', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_image.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_person.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
