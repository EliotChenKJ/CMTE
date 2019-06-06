from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as numpy
from pdb import set_trace as st
import random, os, torch


class MultihalfDataset(BaseDataset):
    """A dataset class for Multihalf image dataset."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        reset some dataset options for half gan.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--K', type=int, default=256, help='sample\'s cropped size')
        parser.add_argument('--n_pic', type=int, default=3, help='number of style pictures')
        parser.set_defaults(preprocess='no', no_flip=True)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """
        Initialize N2 Multihalf dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_dir = os.path.join(self.opt.dataroot, self.opt.phase)
        self.image_paths = sorted(make_dataset(self.image_dir))
        self.image_size = len(self.image_paths) # get the dataset size
        assert self.image_size == self.opt.n_pic
        self.K = self.opt.K

        BtoA = (self.opt.direction == 'BtoA')
        input_nc = self.opt.output_nc if BtoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if BtoA else self.opt.output_nc  
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(self.opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        train_index = index % (self.image_size * 2)
        if self.opt.isTrain:
            # get data_A(Content) / data_B(style) / data_Ref
            if train_index < self.image_size:
                path_Ref = self.image_paths[train_index]        # get a random image path
                path_A = path_Ref
                path_B = path_Ref
                data_Ref = Image.open(path_Ref).convert('RGB')    # needs to be a tensor
                
                # get cropped K * K size sample data_Ref
                w, h = data_Ref.size
                rw = random.randint(0, w - self.K)
                rh = random.randint(0, h - self.K)
                data_Ref = data_Ref.crop((rw, rh, rw + self.K, rh + self.K))
               
                # get cropped 0.5K * 0.5K size sample data_A and data_B
                w, h = data_Ref.size
                rw = random.randint(0, int(w / 2))
                rh = random.randint(0, int(h / 2))
                data_A = data_Ref.crop((rw, rh, int(rw + w / 2), int(rh + h / 2)))
                data_B = data_A.copy()
            else:
                path_A = self.image_paths[train_index % self.image_size]
                data_A = Image.open(path_A).convert('RGB')
                path_B = self.image_paths[(train_index + random.randint(1, self.image_size - 1)) % self.image_size]
                data_B = Image.open(path_B).convert('RGB')
                path_Ref = path_B
                
                # get cropped K * K images from data_Ref
                w, h = data_A.size
                rw = random.randint(0, w - self.K)
                rh = random.randint(0, h - self.K)
                data_A = data_A.crop((rw, rh, rw + self.K, rh + self.K))
                
                wb, hb = data_B.size
                temp_w, temp_h = (min(int(wb * rw / w), wb - self.K), min(int(hb * rh / h), hb - self.K))
                data_Ref = data_B.crop((temp_w, temp_h, temp_w + self.K, temp_h + self.K))
        
                # get cropped 0.5k * 0.5k images from data_B and data_A(K * K)
                w, h = data_Ref.size
                rw = random.randint(0, int(w / 2))
                rh = random.randint(0, int(h / 2))
                data_B = data_Ref.crop((rw, rh, int(rw + w / 2),  int(rh + h / 2)))
                data_A = data_A.crop((rw, rh, int(rw + w / 2),  int(rh + h / 2)))
        else:
            path_A = self.image_paths[train_index % self.image_size]
            data_A = Image.open(path_A).convert('RGB')
            path_B = self.image_paths[train_index % self.image_size]
            data_B = Image.open(path_B).convert('RGB')
            path_Ref = self.image_paths[train_index % self.image_size]
            data_Ref = Image.open(path_Ref).convert('RGB')
            # w, h = data_B.size
            # rw = random.randint(0, w - self.opt.test_size)
            # rh = random.randint(0, h - self.opt.test_size)
            # data_A = data_B.crop((0, 0, w, h))
            # data_B = data_A.crop((0, 0, self.opt.test_size, self.opt.test_size))
            
        # transform data_A and data_B to it standard form 
        data_A = self.transform(data_A)
        data_B = self.transform(data_B)
        data_Ref = self.transform(data_Ref)
        label_style = torch.Tensor(self.opt.n_pic).zero_()
        label_style[train_index % self.image_size] = 1
        
        return {'A': data_A, 'B': data_B, 'Ref': data_Ref,
                'A_paths': path_A, 'B_paths':path_B, 'Ref_paths':path_Ref,
                'label':label_style}

    def __len__(self):
        """Return the total number of images."""
        return self.image_size * 2
    
    def name(self):
        return 'MultihalfDataset'
