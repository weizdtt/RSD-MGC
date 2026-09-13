import argparse
import warnings
from utils.model_trainer import T_trainer

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-name', default='Cub', type=str, help='Data name for training')
parser.add_argument('--seed', default=1, type=int, help='Seed for performance reproduction')
parser.add_argument('-dp', '--data-path', default='./data', type=str, help='Data path for training')
parser.add_argument('-sp', '--save-path', default='./models', type=str, help='Data path for training')
parser.add_argument('-nc', '--n-clusters', default=0, type=int,
                    help='Number of classes, default=0 when getting n_clusters from dataset')
parser.add_argument('-bs', '--batch-size', default=512, type=int, help='Number of batch size')
parser.add_argument('--epochs', default=1000, type=int, help='Number of training epochs')
parser.add_argument('-lr', '--learning-rate', default=0.00001, type=float, help='Learning rate')
parser.add_argument('--device', default=0, type=int, help='Device index for training')
parser.add_argument('-dim', '--dim', default=256, type=int, help='Output dimension of model')
parser.add_argument('-sm', '--save-model', default=False, type=bool, help='Save Model')

parser.add_argument('--steps', default=40, type=int, help='generating steps')
parser.add_argument('-pred', '--pred-v', default='pred_v', type=str, help='generating mode')
parser.add_argument('--vp-rf', default=True, type=bool, help="Boolean flag for vp_rf.")
parser.add_argument("--decoding_rescaling_factor", type=float, default=0.2, help="Factor for rescaling.")
parser.add_argument("--rescaling_factor", type=float, default=0.4, help="Factor for rescaling.")
parser.add_argument("--diffusion_steps", type=int, default=300, help="Number of diffusion steps.")


parser.add_argument('--channels', default=8, type=int, help='number of channels for ortho')
parser.add_argument('--reduce-dim', default=512, type=int, help='dim for AdaptiveAvgPool1d')
parser.add_argument('--iters', default=5, type=int, help='number of iters for IterativeUnit')
parser.add_argument('--num-heads', default=1, type=int, help='number of heads')
parser.add_argument('--use-fusion', default=True, type=bool)
parser.add_argument('--mode-attn', default='mixed', type=str, help='mixed, softmax_only, linear_only')
parser.add_argument('--mode-sfm', default='sa_3', type=str, help='sa_1, sa_2, sa_3')
parser.add_argument('--use-norm', default=True, type=bool)
parser.add_argument('--batch_first', default=True, type=bool)

parser.add_argument('--kv-mean', default=True, type=bool)
parser.add_argument('--p', default=3, type=float)
parser.add_argument('--dropout', default=0.5, type=float)


if __name__ == '__main__':
    args = parser.parse_args()
    model_train = T_trainer(args)
    model_train.running()


