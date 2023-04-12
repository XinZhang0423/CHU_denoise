## Python libraries

# Pytorch
from torch.utils.tensorboard import SummaryWriter

# Math
import numpy as np
import matplotlib.pyplot as plt

# Useful
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

# Local files to import
#from vGeneral import vGeneral
from vDenoising import vDenoising

class iResults(vDenoising):
    def __init__(self,config, *args, **kwargs):
        print("__init__")

    def initializeSpecific(self,config,root, *args, **kwargs):
        # Initialize general variables
        self.initializeGeneralVariables(config,root)
        self.config = config
        #vDenoising.initializeSpecific(self,config,root)

        if ('ADMMLim' in config["method"]):
            self.i_init = 30 # Remove first iterations
            self.i_init = 1 # Remove first iterations
        else:
            self.i_init = 1

        self.defineTotalNbIter_beta_rho(config["method"], config, config["task"])


        # Create summary writer from tensorboard
        self.tensorboard = config["tensorboard"]
        self.writer = SummaryWriter()
        
        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type_im='<f')
        if config["FLTNB"] == "double":
            self.image_gt = self.image_gt.astype(np.float64)

        '''
        image = self.image_gt
        image = image[20,:,:]
        plt.imshow(image, cmap='gray_r',vmin=0,vmax=np.max(image)) # Showing all images with same contrast
        plt.colorbar()
        #os.system('rm -rf' + self.subroot + 'Images/tmp/' + suffix + '/*')
        plt.savefig(self.subroot_data + 'Data/database_v2/' + 'image_gt.png')
        '''

        # Defining ROIs
        self.phantom_ROI = self.get_phantom_ROI(self.phantom)
        if ("3D" not in self.phantom):
            self.bkg_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "background_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            if (self.phantom == "image4_0" or self.phantom == "image4000_0" or self.phantom == "image40_0"):
                self.hot_TEP_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_TEP_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                self.hot_TEP_match_square_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_TEP_match_square_ROI_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                self.hot_perfect_match_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_perfect_match_ROI_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                # This ROIs has already been defined, but is computed for the sake of simplicity
                self.hot_ROI = self.hot_TEP_ROI
            else:
                self.hot_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                # These ROIs do not exist, so put them equal to hot ROI for the sake of simplicity
                self.hot_TEP_ROI = np.array(self.hot_ROI)
                self.hot_TEP_match_square_ROI = np.array(self.hot_ROI)
                self.hot_perfect_match_ROI = np.array(self.hot_ROI)
            self.cold_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')

            # Metrics arrays
            self.PSNR_recon = np.zeros(int(self.total_nb_iter / self.i_init) + 1)
            self.PSNR_norm_recon = np.zeros(int(self.total_nb_iter / self.i_init) + 1)
            self.MSE_recon = np.zeros(int(self.total_nb_iter / self.i_init) + 1)
            self.SSIM_recon = np.zeros(int(self.total_nb_iter / self.i_init) + 1)
            self.MA_cold_recon = np.zeros(int(self.total_nb_iter / self.i_init) + 1)
            self.AR_hot_recon = np.zeros(int(self.total_nb_iter / self.i_init) + 1)
            
            self.AR_hot_TEP_recon = np.zeros(int(self.total_nb_iter / self.i_init) + 1)
            self.AR_hot_TEP_match_square_recon = np.zeros(int(self.total_nb_iter / self.i_init) + 1)
            self.AR_hot_perfect_match_recon = np.zeros(int(self.total_nb_iter / self.i_init) + 1)
            
            self.loss_DIP_recon = np.zeros(int(self.total_nb_iter / self.i_init) + 1)
            self.CRC_hot_recon = np.zeros(int(self.total_nb_iter / self.i_init) + 1)
            self.AR_bkg_recon = np.zeros(int(self.total_nb_iter / self.i_init) + 1)
            self.IR_bkg_recon = np.zeros(int(self.total_nb_iter / self.i_init) + 1)

        if ( 'nested' in self.method or  'Gong' in self.method):
            #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_60it/replicate_' + str(self.replicate) + '/MLEM_it60.img',shape=(self.PETImage_shape),type_im='<d')
            #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'random_1.img',shape=(self.PETImage_shape),type_im='<d')
            #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'F16_GT_' + str(self.PETImage_shape[0]) + '.img',shape=(self.PETImage_shape),type_im='<f')
            try:
                self.image_corrupt = self.fijii_np("/home/xzhang/Documents/DIP/image/BSREM_it30.img",shape=(112,112,1))#(self.subroot_data + 'Data/initialization/' + self.phantom + '/BSREM_30it' + '/replicate_' + str(self.replicate) + '/BSREM_it30.img',shape=(self.PETImage_shape),type_im='<f')
            except:
                self.image_corrupt = self.fijii_np("/home/xzhang/Documents/DIP/image/BSREM_it30.img",shape=(112,112,1))#(self.subroot_data + 'Data/initialization/' + self.phantom + '/BSREM_30it' + '/replicate_' + str(self.replicate) + '/BSREM_it30.img',shape=(self.PETImage_shape),type_im='<d')
            #self.image_corrupt = self.fijii_np("/home/meraslia/workspace_reco/nested_admm/data/Algo/image4_0/replicate_10/nested/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=14_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=1_mlem_=False_A_AML=-100/x_label/24/" + "-1_x_labelconfig_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=14_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=1_mlem_=False_A_AML=-100.img",shape=(self.PETImage_shape))


    def writeBeginningImages(self,suffix,image_net_input=None):
        if (self.tensorboard):
            self.write_image_tensorboard(self.writer,self.image_gt,"Ground Truth (emission map)",suffix,self.image_gt,0,full_contrast=True) # Ground truth in tensorboard
            if (image_net_input is not None):
                self.write_image_tensorboard(self.writer,image_net_input,"DIP input (FULL CONTRAST)",suffix,image_net_input,0,full_contrast=True) # DIP input in tensorboard

    def writeCorruptedImage(self,i,max_iter,x_label,suffix,pet_algo,iteration_name='iterations'):
        if (self.tensorboard):
            if (self.all_images_DIP == "Last"):
                self.write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name,suffix,self.image_gt,i) # Showing all corrupted images with same contrast to compare them together
                self.write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name + " (FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each corrupted image with contrast = 1
            else:       
                if (((max_iter>=10) and (i%(max_iter // 10) == 0)) or (max_iter<10)):
                    self.write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name,suffix,self.image_gt,i) # Showing all corrupted images with same contrast to compare them together
                    self.write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name + " (FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each corrupted image with contrast = 1

    def writeEndImagesAndMetrics(self,i,max_iter,PETImage_shape,f,suffix,phantom,net,pet_algo,iteration_name='iterations'):       
        # Metrics for NN output
        if ("3D" not in phantom):
            self.compute_metrics(PETImage_shape,f,self.image_gt,i,self.PSNR_recon,self.PSNR_norm_recon,self.MSE_recon,self.SSIM_recon,self.MA_cold_recon,self.AR_hot_recon,self.AR_hot_TEP_recon,self.AR_hot_TEP_match_square_recon,self.AR_hot_perfect_match_recon,self.loss_DIP_recon,self.CRC_hot_recon,self.AR_bkg_recon,self.IR_bkg_recon,phantom,writer=self.writer)

        if (self.tensorboard):
            # Write image over ADMM iterations
            if (self.all_images_DIP == "Last"):
                self.write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output)",suffix,self.image_gt,i) # Showing all images with same contrast to compare them together
                self.write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output, FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each image with contrast = 1
                self.write_image_tensorboard(self.writer,f*self.phantom_ROI,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output, FULL CONTRAST CROPPED)",suffix,self.image_gt,i,full_contrast=True) # Showing each image with contrast = 1
            else:
                if (((max_iter>=10) and (i%(max_iter // 10) == 0)) or (max_iter<10)):
                    self.write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output)",suffix,self.image_gt,i) # Showing all images with same contrast to compare them together
                    self.write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output, FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each image with contrast = 1
                    self.write_image_tensorboard(self.writer,f*self.phantom_ROI,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output, FULL CONTRAST CROPPED)",suffix,self.image_gt,i,full_contrast=True) # Showing each image with contrast = 1

    def runComputation(self,config,root):
        if (hasattr(self,'beta')):
            beta_string = ', beta = ' + str(self.beta)

        if (('nested' in config["method"] or  'Gong' in config["method"]) and "results" not in config["task"]):
            self.writeBeginningImages(self.suffix,self.image_net_input) # Write GT and DIP input
            self.writeCorruptedImage(0,self.total_nb_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")
        else:
            self.writeBeginningImages(self.suffix) # Write GT

        if (self.FLTNB == 'float'):
            type_im = '<f'
        elif (self.FLTNB == 'double'):
            type_im = '<d'

        f = np.zeros(self.PETImage_shape,dtype=type_im)
        f_p = np.zeros(self.PETImage_shape,dtype=type_im)

        for i in range(self.i_init,self.total_nb_iter+self.i_init,self.i_init):
            IR = 0
            for p in range(1,self.nb_replicates+1):
                if (config["average_replicates"] or (config["average_replicates"] == False and p == self.replicate)):
                    self.subroot_p = self.subroot_data + 'debug/'*self.debug + '/' + self.phantom + '/' + 'replicate_' + str(p) + '/' + self.method + '/' # Directory root

                    # Take NNEPPS images if NNEPPS is asked for this run
                    if (config["NNEPPS"]):
                        NNEPPS_string = "_NNEPPS"
                    else:
                        NNEPPS_string = ""
                    if ( 'Gong' in config["method"] or  'nested' in config["method"]):
                        if ('post_reco' in config["task"]):
                            pet_algo=config["method"]+"to fit"
                            iteration_name="(post reconstruction)"
                        else:
                            pet_algo=config["method"]
                            iteration_name="iterations"
                        if ('post_reco' in config["task"]):
                            try:
                                f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(-100) + '_epoch=' + format(i-self.i_init) + NNEPPS_string + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
                            except:
                                print("!!!!! failed to read image")
                                break
                        else:
                            f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(i-self.i_init) + "_FINAL" + NNEPPS_string + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
                        if config["FLTNB"] == "double":
                            f_p = f_p.astype(np.float64)
                    elif ('ADMMLim' in config["method"] or config["method"] == 'MLEM' or config["method"] == 'OPTITR' or config["method"] == 'OSEM' or config["method"] == 'BSREM' or config["method"] == 'AML' or 'APGMAP' in config["method"]):
                        pet_algo=config["method"]
                        iteration_name = "iterations"
                        if (hasattr(self,'beta')):
                            iteration_name += beta_string
                        if ('ADMMLim' in config["method"]):
                            subdir = 'ADMM' + '_' + str(config["nb_threads"])
                            subdir = ''
                            #f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_' + format(i) + '_it' + str(config["nb_inner_iteration"]) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                            #f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_' + format(i) + '_it1' + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                            #f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_1'  + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                            f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0'  + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                        #elif (config["method"] == 'BSREM'):
                        #    f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  config["method"] + '_beta_' + str(self.beta) + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                        else:
                            if ('APGMAP' in config["method"]):
                                f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  "APGMAP" + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                            else:
                                f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  config["method"] + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output

                    # Compute IR metric (different from others with several replicates)
                    if ("3D" not in self.phantom):
                        self.compute_IR_bkg(self.PETImage_shape,f_p,int((i-self.i_init)/self.i_init),self.IR_bkg_recon,self.phantom)

                        # Specific average for IR
                        if (config["average_replicates"] == False and p == self.replicate):
                            IR = self.IR_bkg_recon[int((i-self.i_init)/self.i_init)]
                        elif (config["average_replicates"]):
                            IR += self.IR_bkg_recon[int((i-self.i_init)/self.i_init)] / self.nb_replicates
                        
                    if (config["average_replicates"]): # Average images across replicates (for metrics except IR)
                        f += f_p / self.nb_replicates
                    elif (config["average_replicates"] == False and p == self.replicate):
                        f = f_p
                
                    del f_p
                    
            if ("3D" not in self.phantom):
                self.IR_bkg_recon[int((i-self.i_init)/self.i_init)] = IR
                if (self.tensorboard):
                    #print("IR saved in tensorboard")
                    self.writer.add_scalar('Image roughness in the background (best : 0)', self.IR_bkg_recon[int((i-self.i_init)/self.i_init)], i)

            # Show images and metrics in tensorboard (averaged images if asked in config)
            print('Metrics for iteration',int((i-self.i_init)/self.i_init))
            self.writeEndImagesAndMetrics(int((i-self.i_init)/self.i_init),self.total_nb_iter,self.PETImage_shape,f,self.suffix,self.phantom,self.net,pet_algo,iteration_name)

        #self.WMV_plot(config)

    def WMV_plot(self,config):

        self.lr = config['lr']

        # 2.2 plot window moving variance
        plt.figure(1)
        var_x = np.arange(self.windowSize-1, self.windowSize + len(self.VAR_recon)-1)  # define x axis of WMV
        plt.plot(var_x, self.VAR_recon, 'r')
        plt.title('Window Moving Variance,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')  # plot a vertical line at self.epochStar(detection point)
        plt.xticks([self.epochStar, 0, self.total_nb_iter-1], [self.epochStar, 0, self.total_nb_iter-1], color='green')
        plt.axhline(y=np.min(self.VAR_recon), c="black", linewidth=0.5)
        plt.savefig(self.mkdir(self.subroot + '/self.VAR_recon/' + self.suffix + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.VAR_recon-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '.png')

        # Save WMV in tensorboard
        #print("WMV saved in tensorboard")
        for i in range(len(self.VAR_recon)):
            var_x = np.arange(self.windowSize-1, self.windowSize + len(self.VAR_recon)-1)  # define x axis of WMV
            self.writer.add_scalar('WMV in the phantom (should follow MSE trend to find peak)', self.VAR_recon[i], var_x[i])

        # 2.3 plot MSE
        plt.figure(2)
        plt.plot(self.MSE_WMV, 'y')
        plt.title('MSE,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')
        plt.xticks([self.epochStar, 0, self.total_nb_iter-1], [self.epochStar, 0, self.total_nb_iter-1], color='green')
        plt.axhline(y=np.min(self.MSE_WMV), c="black", linewidth=0.5)
        plt.savefig(self.mkdir(self.subroot + '/self.MSE_WMV/' + self.suffix + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.MSE_WMV-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '.png')

        # 2.4 plot PSNR
        plt.figure(3)
        plt.plot(self.PSNR_WMV)
        plt.title('PSNR,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')
        plt.xticks([self.epochStar, 0, self.total_nb_iter - 1], [self.epochStar, 0, self.total_nb_iter - 1], color='green')
        plt.axhline(y=np.max(self.PSNR_WMV), c="black", linewidth=0.5)
        plt.savefig(self.mkdir(self.subroot + '/self.PSNR_WMV/' + self.suffix + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.PSNR_WMV-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '.png')

        #'''
        # 2.5 plot SSIM
        plt.figure(4)
        plt.plot(self.SSIM_WMV, c='orange')
        plt.title('SSIM,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')
        plt.xticks([self.epochStar, 0, self.total_nb_iter - 1], [self.epochStar, 0, self.total_nb_iter - 1], color='green')
        plt.axhline(y=np.max(self.SSIM_WMV), c="black", linewidth=0.5)
        plt.savefig(self.mkdir(self.subroot + '/self.SSIM_WMV/' + self.suffix + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.SSIM_WMV-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '.png')
        #'''
        
        # 2.6 plot all the curves together
        fig, ax1 = plt.subplots()
        fig.subplots_adjust(right=0.8, left=0.1, bottom=0.12)
        ax2 = ax1.twinx()  # creat other y-axis for different scale
        ax3 = ax1.twinx()  # creat other y-axis for different scale
        ax4 = ax1.twinx()  # creat other y-axis for different scale
        ax2.spines.right.set_position(("axes", 1.18))
        p4, = ax4.plot(self.MSE_WMV, "y", label="MSE")
        p1, = ax1.plot(self.PSNR_WMV, label="PSNR")
        p2, = ax2.plot(var_x, self.VAR_recon, "r", label="WMV")
        p3, = ax3.plot(self.SSIM_WMV, "orange", label="SSIM")
        #ax1.set_xlim(0, self.total_nb_iter-1)
        ax1.set_xlim(0, min(self.epochStar+self.patienceNumber,self.total_nb_iter-1))
        plt.title('skip : ' + str(config["skip_connections"]) + ' lr=' + str(self.lr))
        ax1.set_ylabel("Peak Signal-Noise ratio")
        ax2.set_ylabel("Window-Moving variance")
        ax3.set_ylabel("Structural similarity")
        ax4.yaxis.set_visible(False)
        ax1.yaxis.label.set_color(p1.get_color())
        ax2.yaxis.label.set_color(p2.get_color())
        ax3.yaxis.label.set_color(p3.get_color())
        tkw = dict(size=3, width=1)
        ax1.tick_params(axis='y', colors=p1.get_color(), **tkw)
        ax1.tick_params(axis='x', colors="green", **tkw)
        ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
        ax3.tick_params(axis='y', colors=p3.get_color(), **tkw)
        ax1.tick_params(axis='x', **tkw)
        ax1.legend(handles=[p1, p3, p2, p4])
        ax1.axvline(self.epochStar, c='g', linewidth=1, ls='--')
        ax1.axvline(self.windowSize-1, c='g', linewidth=1, ls=':')
        ax1.axvline(self.epochStar+self.patienceNumber, c='g', lw=1, ls=':')
        if self.epochStar+self.patienceNumber > self.epochStar:
            plt.xticks([self.epochStar, self.windowSize-1, self.epochStar+self.patienceNumber], ['\n' + str(self.epochStar) + '\nES point', str(self.windowSize), '+' + str(self.patienceNumber)], color='green')
        else:
            plt.xticks([self.epochStar, self.windowSize-1], ['\n' + str(self.epochStar) + '\nES point', str(self.windowSize)], color='green')
        plt.savefig(self.mkdir(self.subroot + '/combined/' + self.suffix + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+combined-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '.png')

        # 2.4 plot PSNR
        plt.figure(3)
        plt.plot(self.PSNR_WMV)

        '''
        N = 100
        moving_average_PSNR = self.moving_average(self.PSNR_WMV,N)
        plt.plot(np.arange(N-1,len(self.PSNR_WMV)), moving_average_PSNR)
        '''


        plt.title('PSNR,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')
        plt.xticks([self.epochStar, 0, self.total_nb_iter - 1], [self.epochStar, 0, self.total_nb_iter - 1], color='green')
        plt.axhline(y=np.max(self.PSNR_WMV), c="black", linewidth=0.5)
        plt.savefig(self.mkdir(self.subroot + '/self.PSNR_WMV/' + self.suffix + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.PSNR_WMV-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '.png')

    def moving_average(self, series, n):
        # MVA
        ret = np.cumsum(series, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        # EMA
        import pandas as pd
        pd_series = pd.DataFrame(series)
        return pd_series.ewm(com=0.4).mean()

    def compute_IR_bkg(self, PETImage_shape, image_recon,i,IR_bkg_recon,image):
        # radius - 1 is to remove partial volume effect in metrics computation / radius + 1 must be done on cold and hot ROI when computing background ROI, because we want to exclude those regions from big cylinder
        #bkg_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "background_mask" + image[5:] + '.raw', shape=(PETImage_shape),type_im='<f')
        bkg_ROI_act = image_recon[self.bkg_ROI==1]
        #IR_bkg_recon[i] += (np.std(bkg_ROI_act) / np.mean(bkg_ROI_act)) / self.nb_replicates
        IR_bkg_recon[i] = (np.std(bkg_ROI_act) / np.mean(bkg_ROI_act))
        #print("IR_bkg_recon",IR_bkg_recon)
        #print('Image roughness in the background', IR_bkg_recon[i],' , must be as small as possible')

    def compute_metrics(self, PETImage_shape, image_recon,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,SSIM_recon,MA_cold_recon,AR_hot_recon,AR_hot_TEP_recon,AR_hot_TEP_match_square_recon,AR_hot_perfect_match_recon,loss_DIP_recon,CRC_hot_recon,AR_bkg_recon,IR_bkg_recon,image,writer=None):
        # radius - 1 is to remove partial volume effect in metrics computation / radius + 1 must be done on cold and hot ROI when computing background ROI, because we want to exclude those regions from big cylinder
        image_recon_cropped = image_recon*self.phantom_ROI
        image_recon_norm = self.norm_imag(image_recon_cropped)[0] # normalizing DIP output
        image_gt_cropped = image_gt * self.phantom_ROI
        image_gt_norm = self.norm_imag(image_gt_cropped)[0]

        #print('Dif for PSNR calculation',np.amax(image_recon_cropped) - np.amin(image_recon_cropped),' , must be as small as possible')

        # PSNR calculation
        PSNR_recon[i] = peak_signal_noise_ratio(image_gt_cropped, image_recon_cropped, data_range=np.amax(image_recon_cropped) - np.amin(image_recon_cropped)) # PSNR with true values
        PSNR_norm_recon[i] = peak_signal_noise_ratio(image_gt_norm,image_recon_norm) # PSNR with scaled values [0-1]
        #print('PSNR calculation', PSNR_norm_recon[i],' , must be as high as possible')

        # MSE calculation
        MSE_recon[i] = np.mean((image_gt - image_recon)**2)
        """
        if (i >=80):
            plt.figure()
            plt.imshow(image_gt_cropped,cmap='gray')
            plt.figure()
            plt.imshow(image_recon_cropped,cmap='gray')
            plt.show()
        """
        #print('MSE gt', MSE_recon[i],' , must be as small as possible')
        MSE_recon[i] = np.mean((image_gt_cropped - image_recon_cropped)**2)
        #print('MSE phantom gt', MSE_recon[i],' , must be as small as possible')
        
        # SSIM calculation
        SSIM_recon[i] = structural_similarity(np.squeeze(image_gt_cropped), np.squeeze(image_recon_cropped), data_range=(image_recon_cropped).max() - (image_recon_cropped).min())
        #print('SSIM calculation', SSIM_recon[i],' , must be close to 1')

        # Contrast Recovery Coefficient calculation    
        # Mean activity in cold cylinder calculation (-c -40. -40. 0. 40. 4. 0.)
        #cold_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "cold_mask" + image[5:] + '.raw', shape=(PETImage_shape),type_im='<f')
        cold_ROI_act = image_recon[self.cold_ROI==1]
        MA_cold_recon[i] = np.mean(cold_ROI_act)
        if ( 'nested' in self.method or  'Gong' in self.method):
            loss_DIP_recon[i] = np.mean((self.image_corrupt * self.phantom_ROI - image_recon_cropped)**2)
            #loss_DIP_recon[i] = np.sqrt(np.mean((self.image_corrupt * self.phantom_ROI - image_recon_cropped)**2)) / np.max(self.image_corrupt)

        #IR_cold_recon[i] = np.std(cold_ROI_act) / MA_cold_recon[i]
        #print('Mean activity in cold cylinder', MA_cold_recon[i],' , must be close to 0')
        #print('Image roughness in the cold cylinder', IR_cold_recon[i])

        # Mean Activity Recovery (ARmean) in hot cylinder calculation (-c 50. 10. 0. 20. 4. 400)
        #hot_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "tumor_mask" + image[5:] + '.raw', shape=(PETImage_shape),type_im='<f')
        hot_ROI_act = image_recon[self.hot_ROI==1]
        AR_hot_recon[i] = np.mean(hot_ROI_act)
        CRC_hot_recon[i] = np.mean(hot_ROI_act)
        #IR_hot_recon[i] = np.std(hot_ROI_act) / np.mean(hot_ROI_act)
        #print('Mean Activity Recovery in hot cylinder', AR_hot_recon[i],' , must be close to 1')
        #print('Image roughness in the hot cylinder', IR_hot_recon[i])
        
        ### Only useful for new phantom with 3 hot ROIs, but compute it for every phantom for the sake of simplicity ###
        # Mean Activity Recovery (ARmean) in hot cylinder calculation (-c 50. 10. 0. 20. 4. 400)
        hot_TEP_ROI_act = image_recon[self.hot_TEP_ROI==1]
        AR_hot_TEP_recon[i] = np.mean(hot_TEP_ROI_act)
        
        # Mean Activity Recovery (ARmean) in hot cylinder calculation (-c -20. 70. 0. 20. 4. 400)
        hot_TEP_match_square_ROI_act = image_recon[self.hot_TEP_match_square_ROI==1]
        AR_hot_TEP_match_square_recon[i] = np.mean(hot_TEP_match_square_ROI_act)
        
        # Mean Activity Recovery (ARmean) in hot cylinder calculation (-c 50. 90. 0. 20. 4. 400)
        hot_perfect_match_ROI_act = image_recon[self.hot_perfect_match_ROI==1]
        AR_hot_perfect_match_recon[i] = np.mean(hot_perfect_match_ROI_act)

        # Mean Activity Recovery (ARmean) in background calculation (-c 0. 0. 0. 150. 4. 100)
        #m0_bkg = (np.sum(coord_to_value_array(bkg_ROI,image_recon_cropped)) - np.sum([coord_to_value_array(cold_ROI,image_recon_cropped),coord_to_value_array(hot_ROI,image_recon_cropped)])) / (len(bkg_ROI) - (len(cold_ROI) + len(hot_ROI)))
        #AR_bkg_recon[i] = m0_bkg / 100.
        #         
        #bkg_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "background_mask" + image[5:] + '.raw', shape=(PETImage_shape),type_im='<f')
        bkg_ROI_act = image_recon[self.bkg_ROI==1]
        AR_bkg_recon[i] = np.mean(bkg_ROI_act) / 100.
        #IR_bkg_recon[i] = np.std(bkg_ROI_act) / np.mean(bkg_ROI_act)
        #print('Mean Activity Recovery in background', AR_bkg_recon[i],' , must be close to 1')
        #print('Image roughness in the background', IR_bkg_recon[i],' , must be as small as possible')

        del image_recon_cropped
        del image_gt_cropped

        # Save metrics in csv
        from csv import writer as writer_csv
        Path(self.subroot_metrics + self.method + '/' + self.suffix_metrics).mkdir(parents=True, exist_ok=True) # CASToR path
        with open(self.subroot_metrics + self.method + '/' + self.suffix_metrics + '/metrics.csv', 'w', newline='') as myfile:
            wr = writer_csv(myfile,delimiter=';')
            wr.writerow(PSNR_recon)
            wr.writerow(PSNR_norm_recon)
            wr.writerow(MSE_recon)
            wr.writerow(SSIM_recon)
            wr.writerow(MA_cold_recon)
            wr.writerow(AR_hot_recon)
            wr.writerow(AR_hot_TEP_recon)
            wr.writerow(AR_hot_TEP_match_square_recon)
            wr.writerow(AR_hot_perfect_match_recon)
            wr.writerow(AR_bkg_recon)
            wr.writerow(IR_bkg_recon)
            wr.writerow(loss_DIP_recon)
            wr.writerow(CRC_hot_recon)

        '''
        print(PSNR_recon)
        print(PSNR_norm_recon)
        print(MSE_recon)
        print(SSIM_recon)
        print(MA_cold_recon)
        print(AR_hot_recon)
        print(AR_bkg_recon)
        print(IR_bkg_recon)
        '''
        
        if (self.tensorboard):
            print("Metrics saved in tensorboard")
            '''
            writer.add_scalars('MSE gt (best : 0)', {'MSE':  MSE_recon[i], 'best': 0,}, i)
            writer.add_scalars('Mean activity in cold cylinder (best : 0)', {'mean_cold':  MA_cold_recon[i], 'best': 0,}, i)
            writer.add_scalars('Mean Concentration Recovery coefficient in hot cylinder (best : 1)', {'AR_hot':  AR_hot_recon[i], 'best': 1,}, i)
            writer.add_scalars('Mean Concentration Recovery coefficient in background (best : 1)', {'MA_bkg':  AR_bkg_recon[i], 'best': 1,}, i)
            #writer.add_scalars('Image roughness in the background (best : 0)', {'IR':  IR_bkg_recon[i], 'best': 0,}, i)
            '''
            writer.add_scalar('PSNR gt (best : inf)', PSNR_recon[i], i)
            writer.add_scalar('MSE gt (best : 0)', MSE_recon[i], i)
            writer.add_scalar('SSIM gt (best : 0)', SSIM_recon[i], i)
            writer.add_scalar('Mean activity in cold cylinder (best : 0)', MA_cold_recon[i], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in hot cylinder (best : 1)', AR_hot_recon[i], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in hot (only in TEP) cylinder (best : 1)', AR_hot_TEP_recon[i], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in hot (square MR, circle TEP) cylinder (best : 1)', AR_hot_TEP_match_square_recon[i], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in hot (perfect match) cylinder (best : 1)', AR_hot_perfect_match_recon[i], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in background (best : 1)', AR_bkg_recon[i], i)
            writer.add_scalar('DIP loss', loss_DIP_recon[i], i)
            #writer.add_scalar('Image roughness in the background (best : 0)', IR_bkg_recon[i], i)