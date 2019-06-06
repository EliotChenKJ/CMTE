from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as numpy
from pdb import set_trace as st
import random, os


class halfDataset(BaseDataset):
    """A dataset class for half image dataset."""
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
        parser.add_argument('--n_pic', type=int, default=8, help='...')
        parser.set_defaults(batch_size=1, max_dataset_size=20, preprocess='none', no_flip=True)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """
        Initialize Half dataset class.

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

        Step 1: get a random image path and load image data from the disk;
        Step 2: get the k * k data_B and 0.5k * 0.5k data_A images;
        Step 3: convert image data to a PyTorch tensor;
        Step 4: return data points as a dictionary.
        """
        path = self.image_paths[index % self.image_size]   # get a random image path
        data_B = Image.open(path).convert('RGB')    # needs to be a tensor
        if self.opt.isTrain:
            # get cropped K * K size sample data_B
            w, h = data_B.size
            rw = random.randint(0, w - self.K)
            rh = random.randint(0, h - self.K)
            data_B = data_B.crop((rw, rh, rw + self.K, rh + self.K))
            # get cropped 0.5K * 0.5K size sample data_A
            w, h = data_B.size
            rw = random.randint(0, int(w / 2))
            rh = random.randint(0, int(h / 2))
            data_A = data_B.crop((rw, rh, int(rw + w / 2), int(rh + h / 2)))
            # transform data_A and data_B to it standard form    
        else:
            # get cropped pic while testing
            w, h = data_B.size
            # rw = random.randint(0, w - self.opt.test_size)
            # rh = random.randint(0, h - self.opt.test_size)
            data_A = data_B.crop((0, 0, w, h))
            # data_B = data_A.crop((0, 0, self.opt.test_size, self.opt.test_size))
            
        data_A = self.transform(data_A)
        data_B = self.transform(data_B)
        
        return {'A': data_A, 'B': data_B, 
                'A_paths': path, 'B_paths':path}

    def __len__(self):
        """Return the total number of images."""
        return self.image_size
    
    def name(self):
        return 'HalfDataset'
