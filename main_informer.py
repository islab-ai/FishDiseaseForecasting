import argparse
import os
import torch

from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] 사육관리 데이터를 활용한 질병 발생 예측 모델')

## 변경/추가
parser.add_argument('--purpose', type=str, required=True, help='model of experiment')

parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment')
parser.add_argument('--data', type=str, required=True, default='TS_Flatfish', help='data')
parser.add_argument('--data_path', type=str, default='TS_Flatfish.csv', help='data file')
parser.add_argument('--features', type=str, default='MS', help='forecasting task, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='disease_label', help='target feature in MS task')
parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--seq_len', type=int, default=7, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=4, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
parser.add_argument('--enc_in', type=int, default=12, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=12, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.device_ids = '0'

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    
    'TS_Flatfish':{'data':'TS_Flatfish.csv','T':'disease_label','MS':[12,12,1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info['MS']

args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)


## 변경/추가
if args.purpose == 'train':
    Exp = Exp_Informer
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len)
    exp = Exp(args) 
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    torch.cuda.empty_cache()

elif args.purpose == 'test':
    Exp = Exp_Informer
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len)
    exp = Exp(args) # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    torch.cuda.empty_cache()

else:
    print('===== 실행목적을 입력해야합니다. =====')


