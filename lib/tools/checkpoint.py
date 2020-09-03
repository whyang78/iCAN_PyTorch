from pathlib import Path
import torch
from torch import nn,optim
import os
import shutil

class Checkpoint():
    def __init__(self,model,optimizer,ckpt_dir='./checkpoints',max_to_keep=5):
        assert isinstance(model,nn.Module)
        assert isinstance(optimizer,optim.Optimizer)
        assert isinstance(max_to_keep,int) and max_to_keep>=0 #max_to_keep=0时保留全部文件

        self.model=model
        self.optimizer=optimizer

        path=Path(ckpt_dir)
        if path.exists():
            shutil.rmtree(path)
        os.makedirs(path)

        record_pathfile=path/'checkpoints.txt'
        try:
            f=open(record_pathfile,'x')
        except FileExistsError:
            f=open(record_pathfile,'a')
        print(f'checkpoints for path {path}',file=f)
        f.close()

        self.ckpt_path=path
        self.record_path=record_pathfile
        self.max_to_keep=max_to_keep
        self.save_count=0
        self.record_dict={}

    def save(self,ckpt_name=None):
        self.save_count+=1
        save_dict={'model_state_dicts':self.model.state_dict(),
                   'optimizer_state_dicts':self.optimizer.state_dict()}
        save_path=self.ckpt_path/(f'{ckpt_name}' if ckpt_name else f'ckpt_{self.save_count:04d}.pth')
        torch.save(save_dict,save_path)

        f = open(self.record_path, 'a')
        print(f'save_counter:{self.save_count:04d},save_path:{save_path}\n', file=f)
        f.close()

        self.record_dict[self.save_count]=save_path
        if self.save_count>self.max_to_keep and self.max_to_keep>0:
            for count,path in self.record_dict.items():
                if count<=(self.save_count-self.max_to_keep) and path.exists():
                    path.unlink()

    def load(self, load_dir, load_optimizer=True):
        save_dict = torch.load(load_dir)
        self.model.load_state_dict(save_dict['model_state_dicts'])
        if load_optimizer:
            self.optimizer.load_state_dict(save_dict['optimizer_state_dicts'])



