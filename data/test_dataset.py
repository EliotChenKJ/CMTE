from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as numpy
from pdb import set_trace as st
import random, os, torch


class TestDataset(BaseDataset):
    """A dataset class for test image dataset."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        reset some dataset options for test.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--n_pic', type=int, default=3, help='number of style pictures')
        parser.add_argument('--n_cont', type=int, default=10, help='number of content pictures')
        parser.set_defaults(preprocess='no', no_flip=True)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_dir = os.path.join(self.opt.dataroot, self.opt.phase)
        self.image_paths = sorted(make_dataset(self.image_dir))
        print(self.image_paths)
        self.image_size = len(self.image_paths) # get the dataset size
        self.label = []
        for i in range(self.opt.n_cont):
            for j in range(self.opt.n_pic):
                self.label.append((i, j))
        print('Trained images\' labels:', self.label)

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
        train_index = index % len(self.label)
        path_A = self.image_paths[self.label[train_index][0]]
        path_B = self.image_paths[self.label[train_index][1] + self.opt.n_cont]
        data_A = Image.open(path_A).convert('RGB')
        data_B = Image.open(path_B).convert('RGB')
            # w, h = data_B.size
            # rw = random.randint(0, w - self.opt.test_size)
            # rh = random.randint(0, h - self.opt.test_size)
            # data_A = data_B.crop((0, 0, w, h))
            # data_B = data_A.crop((0, 0, self.opt.test_size, self.opt.test_size))
            
        # transform data_A and data_B to it standard form 
        data_A = self.transform(data_A)
        data_B = self.transform(data_B)
        
        return {'A': data_A, 'B': data_B,
                'A_paths': path_A, 'B_paths':path_B}

    def __len__(self):
        """Return the total number of images."""
        return len(self.label)
    
    def name(self):
        return 'MultihalfDataset'
