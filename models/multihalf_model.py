import torch
from .base_model import BaseModel
from . import networks
from .vgg import VGG, GramMatrix, GramMSELoss
import os

class MultihalfModel(BaseModel):
    """
    This is MultihalfModel for image texture extension.

    The model training requires '-dataset_model Multihalf' dataset.
    It trains a half-gan model, mapping from k * k size image to 2k * 2k size image.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(netG='multi_resnet_2x_6blocks', netD='multi_n_layers', n_layers_D=4, 
                            gan_mode='vanilla', pool_size=0, display_ncols=4,
                            niter=50000, niter_decay=50000, save_epoch_freq=10000, display_freq=5000, print_freq=250) 
        if is_train:
            # parser.add_argument('--use_style', type=bool, default=True, help='use style loss')
            parser.add_argument('--lambda_L1', type=float, default=100, help='l1 loss lambda')
            parser.add_argument('--lambda_style', type=float, default=5e3, help='style loss lambda')
        return parser

    def __init__(self, opt):
        """
        Initialize half model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G_GAN', 'G_L1', 'Style', 'G_C']
        # specify the important images. The program will call base_model.get_current_visuals to save and display these images.
        if self.isTrain:
            self.visual_names = ['real_content', 'real_style', 'real_ref', 'fake_ref']
        else:
            self.visual_names = ['real_content', 'real_style', 'fake_ref']
        # specify the models. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.norm, not opt.no_dropout, opt.init_type, 
                                      opt.init_gain, self.gpu_ids)
        if self.isTrain: # if istrain, get netD
            self.netD = networks.define_D(opt.input_nc, opt.ndf, 
                                          opt.netD, opt.n_layers_D, opt.n_pic, opt.norm,
                                          opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # only defined during training time
            # define loss functions. 
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device) # GANLoss
            self.criterionL1 = torch.nn.L1Loss() # L1Loss between fake_B and real_B
            self.criterionBCE = torch.nn.BCELoss()
            # losses of feature map
            self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
            self.style_weights = [self.opt.lambda_style / (t * t) for t in [64, 128, 256, 512, 512]]
            self.criterionStyle = [GramMSELoss().to(self.device)] * len(self.style_layers)
            self.vgg = VGG()
            self.vgg.load_state_dict(torch.load(os.getcwd() + '/models/' + 'vgg_conv.pth'))
            self.set_requires_grad(self.vgg, False)
            self.vgg = self.vgg.to(self.device)
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]
        # show the network structure
        networks.print_network_structure(self.netG)
        if self.isTrain:
            networks.print_network_structure(self.netD)
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.real_content = input['A'].to(self.device)  # get image data A
        self.real_style = input['B'].to(self.device)  # get image data B
        if self.isTrain:
            self.real_ref = input['Ref'].to(self.device)  # get image data B
            self.real_label = input['label'].to(self.device)
            self.real_label = torch.squeeze(self.real_label)
            self.image_paths = input['Ref_paths']  # get image paths

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake_ref = self.netG(self.real_content, self.real_style)  # generate output image given the input data_A
        
    def test(self):
        self.fake_ref = self.netG(self.real_content, self.real_style)

    def backward_D(self):
        """Calculate GAN loss for discriminator"""
        # calculate loss given the input and intermediate results
        pred_fake, _ = self.netD(self.fake_ref.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        pred_real, cT = self.netD(self.real_ref)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        
        self.loss_D_C = self.criterionBCE(cT, self.real_label)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + self.loss_D_C
        self.loss_D.backward()       # calculate gradients of network D w.r.t. loss_D

    def backward_G(self):
        if self.opt.lambda_style != 0:
            style_targets = [GramMatrix()(A).detach() for A in self.vgg(self.real_ref, self.style_layers)]
            out = self.vgg(self.fake_ref, self.style_layers)
            layer_losses = [self.style_weights[a] * self.loss_fns[a](A, style_targets[a]) for a, A in enumerate(out)]
            # print(layer_losses)
            self.style_loss = sum(layer_losses)
            self.style_loss.backward(retain_graph=True)

        self.loss_G_L1 = self.criterionL1(self.fake_ref, self.real_ref) * self.opt.lambda_L1

        # First, G(A) should fake the discriminator
        pred_fake, cF = self.netD(self.fake_ref.clone())
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_C = self.criterionBCE(cF, self.real_label)
        self.loss_G = self.loss_G_GAN + self.loss_G_C + self.loss_G_L1

        self.loss_G.backward()
        
    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()   # clear network G's existing gradients
        self.backward_D()              # calculate gradients for network G
        self.optimizer_D.step()        # update gradients for network G
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()   # clear network G's existing gradients
        self.backward_G()              # calculate gradients for network G
        self.optimizer_G.step()
