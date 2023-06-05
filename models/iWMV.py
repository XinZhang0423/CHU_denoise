from torch import from_numpy
from numpy import inf, zeros, float64, squeeze, newaxis, ones_like, amax, amin, mean, ones
from numpy.linalg import norm
import numpy as np
from utils.pre_utils import *
class iWMV():
    def __init__(self, config):
        self.queueQ = []
        self.VAR_min = inf
        self.SUCCESS = False
        self.stagnate = 0

    def initializeSpecific(self,config,root, *args, **kwargs):
        self.patienceNumber = config["patienceNumber"]
        self.epochStar = -1
        self.VAR_recon = []
        self.MSE_WMV = []
        self.PSNR_WMV = []
        self.SSIM_WMV = []
        self.SUCCESS = False

        self.EMV_or_WMV = config["EMV_or_WMV"]
        if self.EMV_or_WMV == "EMV":    
            self.EMA = zeros((self.PETImage_shape))
            self.EMV = 0
            self.alpha_EMV = config["alpha_EMV"]
        else:
            self.windowSize = config["windowSize"]

        #self.queueQ = array((self.windowSize,self.PETImage_shape))

        #Loading Ground Truth image to compute metrics
        # self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type_im='<f')
        groundtruth_path = np.load("/home/xzhang/Documents/我的模型/data/ground_truth/ground_truth.npy")
        self.PETImage_shape = (112,112,1)
        self.image_gt = groundtruth_path
        if config["FLTNB"] == "double":
            self.image_gt = self.image_gt.astype(float64)

        # Load phantom ROI
        if (self.PETImage_shape[2] == 1): # 2D
            self.phantom_ROI = np.load("/home/xzhang/Documents/我的模型/data/corrupted_images/BSREM_it30.npy")
            # self.phantom_ROI = self.fijii_np('/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/database_v2/image40_0/phantom_mask40_0.raw',shape=(self.PETImage_shape),type_im='<f')
        else: # 3D
            self.phantom_ROI = ones(self.PETImage_shape)

    def runComputation(self,config,root):
        pass

    def WMV(self,out,epoch,queueQ,SUCCESS,VAR_min,stagnate):
        
        # Descale, squeeze image and add 3D dimension to 1 (ok for 2D images)
        out = descale_imag(from_numpy(out),self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
        out = squeeze(out)
        if (len(out.shape) == 2): # 2D
            out = out[:,:,newaxis]
        else: # 3D
            out = out.reshape(out.shape[::-1])

        out_cropped = out * self.phantom_ROI
        image_gt_cropped = self.image_gt * self.phantom_ROI

        from skimage.metrics import peak_signal_noise_ratio
        from skimage.metrics import structural_similarity
        self.MSE_WMV.append(mean((image_gt_cropped - out_cropped)**2))
        self.PSNR_WMV.append(peak_signal_noise_ratio(image_gt_cropped, out_cropped, data_range=amax(out_cropped) - amin(out_cropped)))
        self.SSIM_WMV.append(structural_similarity(squeeze(image_gt_cropped), squeeze(out_cropped), data_range=out_cropped.max() - out_cropped.min()))

        if (self.EMV_or_WMV == "WMV"):
            #'''
            #####################################  Window Moving Variance  #############################################
            queueQ.append(out_cropped.flatten()) # Add last computed image to last element in queueQ from window
            if (len(queueQ) == self.windowSize):
                # Compute mean for this window
                mean_im = queueQ[0].copy()
                for x in queueQ[1:self.windowSize]:
                    mean_im += x
                mean_im = mean_im / self.windowSize
                # Compute variance for this window
                VAR = norm(queueQ[0] - mean_im) ** 2
                for x in queueQ[1:self.windowSize]:
                    VAR += norm(x - mean_im) ** 2
                VAR = VAR / self.windowSize
                # Check if current variance is smaller than minimum previously computed variance, else count number of iterations since this minimum
                if VAR < VAR_min and not SUCCESS:
                    VAR_min = VAR
                    self.epochStar = epoch  # current detection point
                    stagnate = 1
                else:
                    stagnate += 1
                # ES point has been found
                if stagnate == self.patienceNumber:
                    SUCCESS = True
                queueQ.pop(0) # Remove first element in queueQ from window for next variance computation
                self.VAR_recon.append(VAR) # Store current variance to plot variance curve after
            #'''
        else:
            #'''
            #####################################  Exponential Moving Variance  #############################################
            # Compute variance for this window
            self.EMV = (1-self.alpha_EMV) * (self.EMV + self.alpha_EMV * norm(out_cropped - self.EMA)**2)
            # Compute EMA to be used in next window
            self.EMA = (1-self.alpha_EMV) * self.EMA + self.alpha_EMV * out_cropped
            # Check if current variance is smaller than minimum previously computed variance, else count number of iterations since this minimum
            if self.EMV < VAR_min and not SUCCESS:
                VAR_min = self.EMV
                self.epochStar = epoch  # current detection point
                stagnate = 1
            else:
                stagnate += 1
            # ES point has been found
            if stagnate == self.patienceNumber:
                SUCCESS = True
            self.VAR_recon.append(self.EMV) # Store current variance to plot variance curve after
            #'''


        #'''
        if SUCCESS:
            # Open output corresponding to epoch star
            # net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(self.epochStar) + '.img'
            net_outputs_path = "/home/xzhang/Documents/我的模型/data/results/images/es/output_" + format(self.epochStar) + ".img"
            out = fijii_np(net_outputs_path,shape=(self.PETImage_shape),type_im='<f')
            
            # Descale like at the beginning
            out = descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
            #out = self.descale_imag(from_numpy(out),self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)

            # Saving ES point image
            # net_outputs_path = self.subroot + 'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/ES_out_' + self.net +  str(self.global_it) + '_epoch=' + format(self.epochStar) + '.img'
            net_outputs_path = "/home/xzhang/Documents/我的模型/data/results/images/es/ES_output_" + format(self.epochStar) + ".img"
            with open(net_outputs_path,'wb') as fp:
                out.tofile(fp)

            print("#### WMV ########################################################")
            print("                 ES point found, epoch* =", self.epochStar)
            print("#################################################################")
        #'''

        return SUCCESS, VAR_min, stagnate