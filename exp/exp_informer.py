from data.data_loader import Dataset_Custom
from exp.exp_basic import Exp_Basic
from models.model import Informer

from utils.tools import EarlyStopping
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryConfusionMatrix

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import json

import warnings
warnings.filterwarnings('ignore')
import random
import torch.backends.cudnn as cudnn
import datetime

# seed 고정
seed = 20230111
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)

# 평가용 log 파일 생성 및 json으로 저장
class LogSaver:
    def __init__(self, args):
        self.test = dict()
        self.meta = dict()
        self.metrics = dict()
        self.current_index = 1
        
        ## 변경/추가
        command = "python -u main_informer.py --purpose test --model informer --data TS_Flatfish"
        
        for key in args.keys():
            command += f" --{key} {args[key]}"
        
        self.update_meta('command', command)
            
    def update_meta(self, key, value):
        self.meta[key] = value;
    
    def configure_time(self, flag='start'):
        for i in range(2):
            t = datetime.datetime.now()
            result = t.strftime("%Y-%m-%d %H:%M:%S")
            if flag == 'start':
                key = f'start_time_{i}'
            if flag == 'end':
                key = f'end_time_{i}'
            self.update_meta(key, result)
    
    def update_test(self, input_data, pred, true):
        temp = dict() 
        for _list in input_data:
            for j in _list:
                index = _list.index(j)
                if index == 5 and _list[index] != 1:
                    _list[index] = round(_list[index])
                if index == 9 and _list[index] != 1:
                    _list[index] = round(_list[index])
        temp['input'] = input_data
        temp['pred'] = pred
        temp['true'] = true
        self.test[self.current_index] = temp
        self.current_index += 1
        
    def update_metrics(self, value, flag='conf_mat'):
        if flag == 'conf_mat':
            self.metrics['confusion_matrix'] = value.to('cpu').detach().numpy().tolist()
        if flag == 'f1_score':
            self.metrics['f1-score'] = value.to('cpu').detach().numpy().tolist()
        if flag == 'acc':
            self.metrics['accuracy'] = value.to('cpu').detach().numpy().tolist()
        
    def save_all(self, log_filename):
        result_dict = dict()
        result_dict['meta'] = self.meta
        result_dict['test'] = self.test
        result_dict['metrics'] = self.metrics
        
        with open(log_filename, 'w') as file: 
            json.dump(result_dict, file, indent=4)

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
        }
        if self.args.model == 'informer':
            e_layers = 3 
            model =  Informer(  
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out,
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len).to(self.device)
            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'TS_Flatfish':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0
        
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = 1; freq='d'
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq='d'
        data_set = Data(
            root_path='./dataset/',
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=False,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=1e-5)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.BCELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            # preds.extend(pred)
            true = vali_data.inverse_transform(true)
            # trues.extend(true.flatten().detach().cpu().numpy())
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=4, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                
                true = train_data.inverse_transform(true)
                
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            
            with torch.no_grad():
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)
                early_stopping(vali_loss, self.model, path)
            
            print("Epoch: {0} | Train Loss: {1:.7f} Valid Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def test(self, setting):
        
        ## 저장한 가중치 load
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        test_data, test_loader = self._get_data(flag='test')
        metrics = BinaryF1Score().to(self.device)
        conf_mat = BinaryConfusionMatrix()
        binary_acc = BinaryAccuracy()
        count=0
        self.model.eval()
        
        raw_preds = []
        preds = []
        trues = []
        f1 = []
        Logfile = LogSaver(args=dict())
        Logfile.configure_time(flag='start')
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            
            raw_preds.extend(pred)
            
            # logfile 저장용 input 추출
            input_data = test_data.inverse_transform(batch_x)
            input_data = input_data.numpy().tolist()
            input_data = sum(input_data, [])

            true = test_data.inverse_transform(true)
            pred = torch.abs(torch.round(pred))            

            preds.extend(pred.flatten().detach().cpu().numpy())
            trues.extend(true.flatten().detach().cpu().numpy())
            
            save_pred = sum(pred.to('cpu').detach().numpy().tolist(), [])
            save_true = sum(true.to('cpu').numpy().tolist(), [])
            
            Logfile.update_test(input_data, sum(save_pred, []), sum(save_true, []))
            
        for p, t in zip(preds, trues):
            if p == t:
                count+=1
        
        tensor_rawpreds = torch.tensor(raw_preds)
        tensor_preds = torch.tensor(np.array(preds))
        tensor_trues = torch.tensor(np.array(trues))
        print('correct_count:', count)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        acc = binary_acc(tensor_rawpreds, tensor_trues)
        confusion_mat = conf_mat(tensor_preds, tensor_trues).to(self.device)
        f1score = metrics(tensor_preds, tensor_trues)

        print('Test_Accuracy:{:0.7f}, F1-Score:{:0.7f}'.format(acc, f1score))
        
        # 로그 저장
        Logfile.configure_time(flag='end')
        Logfile.update_metrics(confusion_mat, flag='conf_mat')
        Logfile.update_metrics(f1score, flag='f1_score')
        Logfile.update_metrics(acc, flag='acc')
        Logfile.save_all('test_logfile.json')

        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)

        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        f_dim = -1
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y