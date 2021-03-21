 #!/usr/bin/env python
# -*- coding: utf-8 -*-


#マスク有　切り抜き有　ネットワーク　：　ResNet7層

from __future__ import absolute_import
from six.moves import range

import os
import time
import numpy as np
from scipy import stats
from tqdm import tqdm

#   NNabla
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context  # GPU

#   Image
import cv2

#   Originals
from settings_New import settings
import data_New as dt

# plot
import matplotlib.pyplot as plt


# -------------------------------------------
#   Network for IQA
# ------------------------------------------
def network(input, feature_num, scope="", ch=2):
    """
    Define Convolutional DNN
       input:   input image ( size must be 64 by 64 ) = 3 x 64 x 64 Tensor
       test:    test or not
    """

    def Residual_Block(x, n_out, n_middle, name=''):
        r = F.relu(PF.convolution(x, n_middle, kernel=(1, 1), name=name +'1'))
        r = F.relu(PF.convolution(r, n_middle, kernel=(3, 3), pad=(1, 1), name=name + '2'))
        r = PF.convolution(r, n_out, kernel=(1, 1), name=name + '3')
        return F.relu(r + x)

    def convblock(x, n, f_size=3, name=''):  # 畳み込み　x:入力　n:出力のチャンネル数(=フィルタの数)　f_size:kernelフィルタのサイズ9*9(重み)
        r = PF.convolution(x, n, kernel=(f_size, f_size), pad=(f_size // 2, f_size // 2), stride=(2, 2),
                           name=name)  # paddingは基本フィルタサイズの半分（切り捨て）　strideフィルタのずらし幅　name入れ物の名前
        return F.relu(r)  # 活性化関数？

    with nn.parameter_scope(scope):  # 64x64conv  with - nn.parameter-scopeのオブジェクトを用意

        # (3, 256, 256) -> (256, 256, 256) -> (256, 128, 128)
        #c1 = convblock(input, 2 * (ch * 3), name='cnv1')  # input 入力画像　128フィルタの数
        #c2 = convblock(c1, 4 * (ch * 3), name='cnv2')   # input 入力画像　128フィルタの数
        #c3 = convblock(c2, 8 * (ch * 3), name='cnv3')   # input 入力画像　128フィルタの数
        #c4 = convblock(c3, 16 * (ch * 3), name='cnv4')  # input 入力画像　128フィルタの数

        c1 = Residual_Block(input, ch*3, ch, name='res1')
        c1 = convblock(c1, 2 * (ch*3), name='conv1')

        c2 = Residual_Block(c1, 2*  (ch*3), 2*ch, name='res2')
        c2 = convblock(c2, 4 * (ch * 3), name='conv2')

        c3 = Residual_Block(c2, 4 * (ch * 3), 4 * ch, name='res3')
        c3 = convblock(c3, 8 * (ch * 3), name='conv3')

        c4 = Residual_Block(c3, 8 * (ch * 3), 8 * ch, name='res4')
        c4 = convblock(c4, 16 * (ch * 3), name='conv4')

        c5 = Residual_Block(c4, 16 * (ch * 3), 16 * ch, name='res5')
        c5 = convblock(c5, 32 * (ch * 3), name='conv5')

        c6 = Residual_Block(c5, 32 * (ch * 3), 16 * ch, name='res6')
        c6 = convblock(c6, 64 * (ch * 3), name='conv6')

        c7 = Residual_Block(c6, 64 * (ch * 3), 16 * ch, name='res7')
        c7 = convblock(c7, 128 * (ch * 3), name='conv7')

        #c8 = Residual_Block(c7, 128 * (ch * 3), 16 * ch, name='res8')
        #c8 = convblock(c8, 256 * (ch * 3), name='conv8')

        c_ap= F.mean(c7, axis=(2, 3))                    # Global Average Pooling (96, 256, 256, 256) > (96)

        # Affine Layer : 256 -> batch,feature_num
        c_out = F.relu(PF.affine(c_ap, (c_ap.shape[1]//2,), name='Affine'))
        c_out = F.relu(PF.affine(c_out, (feature_num,), name='Affine2'))

    return c_out


def network2(input, scope=""):  # Fullconnectnetwork

    with nn.parameter_scope(scope):
        # Affine Layer : 16,8,8 -> 128
        # c5 = F.leaky_relu(PF.affine(input, (256,), name='Affine1'), 0.01)

        # Affine Layer : M -> 1
        c5 = PF.affine(input, (1,), name='Affine_out')

        return c5


# -------------------------------------------
#   Training
# -------------------------------------------
def train(args):
    """
    Training
    """

    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Initial settings
    ##  ~~~~~~~~~~~~~~~~~~~

    #   Input Variable　      args. -> setting.
    M = 64

    nn.clear_parameters()  # Clear　
    Input = nn.Variable([args.batch_size, 6, 256, 256])
    Trues = nn.Variable([args.batch_size, 1])  # True Value

    #   Network Definition
    Name = "CNN"  # Name of scope which includes network models (arbitrary)
    Name2 = "CNN"
    preOutput = network(input=Input, feature_num=M, scope=Name)    # Network & Output #add
    preOutput = F.reshape(preOutput, (args.batch_size, 1, M))      # (B*N, M) > (B, N, M)
    preOutput = F.mean(preOutput, axis=1, keepdims=True)  # (B, N, M) > (B, 1, M) N個のシフト画像の特徴量を１つにする keepdims->次元を保持
    Output = network2(input=preOutput, scope=Name2)  # fullconnect

    #   Loss Definition
    Loss = F.mean(F.absolute_error(Output, Trues))  # Loss Function (Squared Error) 誤差関数(差の絶対値の平均）　-> 交差エントロピーはだめ？

    #   Solver Setting
    solver = S.Adam(args.learning_rate)  # Adam is used for solver　学習率の最適化
    solver2 = S.Adam(args.learning_rate)  # Adam is used for solver　学習率の最適化
    solver.weight_decay(0.00001)  # Weight Decay for stable update
    solver2.weight_decay(0.00001)

    with nn.parameter_scope(Name):  # Get updating parameters included in scope
        solver.set_parameters(nn.get_parameters())

    with nn.parameter_scope(Name2):  # Get updating parameters included in scope
        solver2.set_parameters(nn.get_parameters())

    #   Training Data Setting
    #image_data, mos_data, image_files = dt.data_loader(test = False)
    image_data, mos_data  = dt.data_loader(test=False)

    #batches = dt.create_batch(image_data, mos_data, args.batch_size, image_files)
    batches = dt.create_batch(image_data, mos_data, args.batch_size)
    del image_data, mos_data

    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Learning
    ##  ~~~~~~~~~~~~~~~~~~~
    print('== Start Training ==')

    bar = tqdm(total=(args.epoch - args.retrain)*batches.iter_n, leave=False)
    bar.clear()
    cnt = 0
    loss_disp = True

    #   Load data
    if args.retrain > 0:  # 途中のエポック(retrain)から再学習
        with nn.parameter_scope(Name):
            print('Retrain from {0} Epoch'.format(args.retrain))
            nn.load_parameters(os.path.join(args.model_save_path, "network_param_{:04}.h5".format(args.retrain)))
            solver.set_learning_rate(args.learning_rate / np.sqrt(args.retrain))

    ##  Training
    for i in range(args.retrain, args.epoch):  # args.retrain → args.epoch まで繰り返し学習

        bar.set_description_str('Epoch {0}/{1}:'.format(i + 1, args.epoch), refresh=False)  # プログレスバーに説明文を加える

        #   Shuffling
        batches.shuffle()

        ##  Batch iteration
        for j in range(batches.iter_n):  # バッチ学習

            cnt += 1

            #  Load Batch Data from Training data
            Input_npy, Trues_npy = next(batches)
            size_ = Input_npy.shape
            Input.d = Input_npy.reshape([size_[0]*size_[1], size_[2], size_[3], size_[4]])
            Trues.d = Trues_npy

            #  Update
            solver.zero_grad()  # Initialize　 #   Initialize #勾配をリセット
            #solver2.zero_grad()
            Loss.forward(clear_no_need_grad=True)  # Forward path　#順伝播
            loss_scale = 8
            Loss.backward(loss_scale, clear_buffer=True)  # Backward path　#誤差逆伝播法
            #solver2.update()
            solver.scale_grad(1. / loss_scale)
            solver.update()

            # Progress
            if cnt % 10 == 0:
                bar.update(10)  # プログレスバーの進捗率を1あげる
                if loss_disp is not None:
                    bar.set_postfix_str('Loss={0:.3e}'.format(Loss.d), refresh=False)  # 実行中にloss_dispとSRCCを表示

        ## Save parameters
        if ((i + 1) % args.model_save_cycle) == 0 or (i + 1) == args.epoch:
            bar.clear()
            with nn.parameter_scope(Name):
                nn.save_parameters(os.path.join(args.model_save_path, 'network_param_{:04}.h5'.format(i + 1)))
            with nn.parameter_scope(Name2):
                nn.save_parameters(os.path.join(args.model_save_path2, 'network_param_{:04}.h5'.format(i + 1)))


# -------------------------------------------
#   Test
# -------------------------------------------
def test(args):
    """
    Training
    """
    M = 64
    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Initial settings
    ##  ~~~~~~~~~~~~~~~~~~~

    #   Input Variable　変数定義
    nn.clear_parameters()  # Clear
    Input = nn.Variable([1, 6, 256, 256])  # Input
    Trues = nn.Variable([1, 1])  # True Value

    #   Network Definition
    Name = "CNN"  # Name of scope which includes network models (arbitrary)
    Name2 = "CNN"

    preOutput = network(input=Input, feature_num=M, scope=Name)  # Network & Output #add
    preOutput = F.reshape(preOutput, (1, N, M))  # (B*N, M) > (B, N, M)
    preOutput_mean = F.mean(preOutput, axis=1,
                            keepdims=True)  # (B, N, M) > (B, 1, M) N個のシフト画像の特徴量を１つにする keepdims->次元を保持
    Output_test = network2(input=preOutput_mean, scope=Name2)  # fullconnect

    Loss_test = F.mean(F.absolute_error(Output_test, Trues))  # Loss Function (Squared Error) #誤差関数

    #   Load data　保存した学習パラメータの読み込み
    with nn.parameter_scope(Name):
        nn.load_parameters(os.path.join(args.model_save_path, "network_param_{:04}.h5".format(args.epoch)))
    with nn.parameter_scope(Name2):
        nn.load_parameters(os.path.join(args.model_save_path2, "network_param_{:04}.h5".format(args.epoch)))

    # Test Data Setting
    #image_data, mos_data, image_files = dt.data_loader(test=True)
    image_data, mos_data= dt.data_loader(test=True)
    #batches = dt.create_batch(image_data, mos_data, 1, image_files)
    batches = dt.create_batch(image_data, mos_data, 1)

    del image_data, mos_data

    truth = []
    result = []

    for j in range(batches.iter_n):
        #Input_npy, Trues_npy, image_files = next(batches)
        Input_npy, Trues_npy = next(batches)
        size_ = Input_npy.shape
        # print("Input Image:" +  str(image_files) + " Trues:" + str(Trues_npy))
        Input.d = Input_npy.reshape([size_[0] *  size_[1] , size_[2], size_[3], size_[4]])
        Trues.d = Trues_npy[0][0]

        Loss_test.forward(clear_no_need_grad=True)
        result.append(Loss_test.d)
        truth.append(Trues.d)

    result = np.array(result)
    mean = np.mean(result)
    truth = np.squeeze(np.array(truth))  # delete

    # Evaluation of performance
    mae = np.average(np.abs(result - truth))
    SRCC, p1 = stats.spearmanr(truth, result)  # Spearman's Correlation Coefficient
    PLCC, p2 = stats.pearsonr(truth, result)

    np.set_printoptions(threshold=np.inf)
    print("result: {}".format(result))
    print("Trues: {}".format(truth))
    print(np.average(result))
    print("\n Model Parameter [epoch={0}]".format(args.epoch))
    print(" Mean Absolute Error with Truth: {0:.4f}".format(mae))
    print(" Speerman's Correlation Coefficient: {0:.5f}".format(SRCC))
    print(" Pearson's Linear Correlation Coefficient: {0:.5f}".format(PLCC))
    #os.remove("./pkl/test_eval.pkl")  # add
    #os.remove("./pkl/test_image.pkl")  # add


# -------------------------------------------
#   Real-Time Demo
# -------------------------------------------
def demo(args):
    """
    Training
    """

    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Arbitrary Parameters
    ##  ~~~~~~~~~~~~~~~~~~~
    #   Evaluation Settings
    Frame_per_calc = 10  # No. of Frames per Calculation Cycle
    magnification = 1.2  # Fine tuning for score value

    #   Video Settings
    frame_rate = 25.0  # Frame per Second
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # Video file format (mp4)

    #   Display Settings
    font_type = cv2.FONT_HERSHEY_SIMPLEX  # Font Type : Hershey fonts
    front_color = (0, 0, 0)  # Font Color : front
    back_color = (255, 255, 255)  # Font Color : background
    position = (30, 50)  # Context Position

    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Initial settings
    ##  ~~~~~~~~~~~~~~~~~~~

    #   Input Variable
    nn.clear_parameters()  # Clear
    Input = nn.Variable([1, 3, 64, 64])  # Input

    #   Network Definition
    Name = "CNN"  # Name of scope which includes network models (arbitrary)
    Name2 = "CNN"
    Output = network(Input, scope=Name)  # Network & Output

    #   Load data
    with nn.parameter_scope(Name):
        print(args.epoch)
        nn.load_parameters(os.path.join(args.model_save_path, "network_param_{:04}.h5".format(args.epoch)))
    with nn.parameter_scope(Name2):
        print(args.epoch)
        nn.load_parameters(os.path.join(args.model_save_path2, "network_param_{:04}.h5".format(args.epoch)))

    #   Video Device
    deviceID = 0
    cap = cv2.VideoCapture(deviceID)

    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Real-time IQA
    ##  ~~~~~~~~~~~~~~~~~~~
    #   Get video information
    _, frame = cap.read()  # Capture video at once
    height = frame.shape[0]  # Video size : height
    width = frame.shape[1]  # Video size : width
    if height > width:
        trim_height = round(abs(height - width) / 2)
        trim_width = 0
    else:
        trim_height = 0
        trim_width = round(abs(height - width) / 2)

    #   Temporary Parameters for calculation
    cnt = 0
    result = []
    result_ave = 0
    video_coding = 0

    while (True):

        # Capture from Video device
        ret, frame = cap.read()

        #   Waiting keyboad input
        key = cv2.waitKey(40) & 0xFF

        #   Resizing Image
        frame_trim = frame[trim_height:height - trim_height, trim_width:width - trim_width,
                     :]  # Trimming so as to be square size
        frame_resize = cv2.resize(frame_trim, (64, 64), interpolation=cv2.INTER_AREA).transpose(2, 0,
                                                                                                1)  # Resize (*,*,3) -> (3,64,64)

        #   Processing
        Input.d = np.expand_dims(frame_resize, axis=0)  # Add axis to match input (3,64,64) -> (1,3,64,64)
        Output.forward()

        #   Storing Score
        score = np.max([min([Output.d[0][0] / 9 * 100, 100]), 0])  # 1~9の評点を0~100に換算
        result.append(score)

        #   Averaging Score
        if cnt > Frame_per_calc:
            #   Average Storing Score
            result_ave = (np.average(np.array(result)))
            result_ave = np.max([np.min([magnification * result_ave, 100]), 0])  # fine tuning

            #   Just for check
            # print('  IQA Value  :: {0:.1f}/{1}'.format(result_ave, 100))

            # Initialization
            cnt = 0
            result = []

        cnt += 1

        # v : Start to save video
        if key == ord('v'):
            writer = cv2.VideoWriter('result/video.mp4', fmt, frame_rate, (width, height))
            video_coding = 1

        # t : Stop to save video
        if key == ord('t'):
            video_coding = 0
            try:
                writer.release()
            except:
                pass

        # q : Exit
        if key == ord('q'):
            try:
                writer.release()
            except:
                pass
            break

        #   Display image
        txt_ = 'Score : {0:.0f}%'.format(result_ave)
        cv2.putText(frame, txt_, position, font_type, 1.2, back_color, 5, cv2.LINE_AA)
        cv2.putText(frame, txt_, position, font_type, 1.2, front_color, 1, cv2.LINE_AA)
        Img_disp = cv2.resize(frame, (round(width * 1.5), round(height * 1.5)), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('frame', Img_disp)

        #   Save Video
        if video_coding:
            writer.write(frame)

    #   Finish Capturing
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    Demo = False  # If demo, set "True".　　

    Test = False    # If test, set "True". Otherwise training, set "False".

    if not Demo:
        # GPU connection
        ctx = get_extension_context('cudnn', device_id=0, type_config="half")
        nn.set_default_context(ctx)
        #   Train
        if not Test:
            train(settings())
        else:
            test(settings())
    else:
        #   Demo
        demo(settings())
