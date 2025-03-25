from argparse import ArgumentParser

parser=ArgumentParser()

# Global
parser.add_argument('--gpu', type=str, dest='gpu', default='0')

# For Meta-test
parser.add_argument('--inputpath', type=str, dest='inputpath', default=r'D:\元迁移学习\实验\Data\裁剪500\Loess\Loess')
parser.add_argument('--gtpath', type=str, dest='gtpath', default=r'D:\元迁移学习\实验\Data\裁剪500\Loess\Loess')
#parser.add_argument('--kernelpath', type=str, dest='kernelpath', default=r'D:\Edge_download\MZSR-master\MZSR-master\Input\g20\kernel.mat')
parser.add_argument('--savepath', type=str, dest='savepath', default=r'D:\元迁移学习\实验\不同模型进行元迁移学习\TfaSR元迁移学习\Code\metalearning\result')
parser.add_argument('--model', type=int, dest='model', choices=[0,1,2,3], default=0)
parser.add_argument('--num', type=int, dest='num_of_adaptation', choices=[1,10], default=1)

# For Meta-Training
parser.add_argument('--trial', type=int, dest='trial', default=0)
parser.add_argument('--step', type=int, dest='step', default=0)
# false是test， true是train
parser.add_argument('--train', dest='is_train', default=True, action='store_true')

args= parser.parse_args()

#Transfer Learning From Pre-trained model.
IS_TRANSFER = False
TRANS_MODEL = 'Pretrained/Pretrained'

# Dataset Options
HEIGHT=64
WIDTH=64
CHANNEL=3

# SCALE_LIST=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
SCALE_LIST=[2.0]

META_ITER=100000
META_BATCH_SIZE=4
META_LR=1e-4

TASK_ITER=4
TASK_BATCH_SIZE=8
TASK_LR=1e-2

# Loading tfrecord and saving paths
TFRECORD_PATH='train_SR_MZSR.tfrecord'
# tfasr
CHECKPOINT_DIR=r"D:\元迁移学习\实验\不同模型进行元迁移学习\TfaSR元迁移学习\Code\metalearning\tfasr_99\tfasr_099.pth"
