 #
# Settings for SEGAN
#

class settings:

    def __init__(self):

        #   Precision Mode
        self.halfprec = True                        # 16bit or not

        #   Image settings
        #self.size               = (64,64)           # Input Size (64 by 64)
        #self.size = (128, 128)                           # Input Size (128 by 128)
        #self.size = (256, 256)                         #Input Size (256 by 256)
        #   Training
        self.batch_size = 50                       # Batch size
        self.batch_size_test = 1
        self.epoch      = 500 # Epoch
        self.learning_rate = 0.00001                # Learning Rate　　

        # Retrain
        self.retrain    = 0

        # Save path
        self.model_save_path    = 'params'          # Network model path
        self.model_save_path2 = 'params2'  # Network model path
        self.model_save_cycle   = 100               # Epoch cycle for saving model (init:1)
        self.result_save_path   = 'result'          # Network model path

        # Data path
        self.train_data_path    = './resize/eximages_128x128'     # Folder containing training image (train)
        self.test_data_path     = './data/brother_girl'   # Folder containing test image (test)
        self.train_mask_path     = './resize/exreference_128x128'
        self.test_mask_path = './resize/exreference_128x128'
        # Pkl files
        self.pkl_path     = 'pkl'             # Folder of pkl files for train
'''
1:5.441364765167236
10:5.437655448913574
20:5.42940616607666
30:5.4213080406188965
40:5.416585922241211
50:5.414455413818359 
60:5.400368690490723
70:5.385716438293457
80:5.389223575592041
90:5.373013973236084
100:5.37229061126709
200:5.288339614868164
400:5.185093402862549
600:5.068819046020508
800:4.983580589294434
1000:4.82247257232666
1200:4.742821216583252
1400:4.69171667098999
1600:4.5897722244262695
1800:4.513787269592285
'''
