import torch
import numpy as np
import os

class Logger_detect(object):
    """ logger for ood detection task"""
    def __init__(self, args,save_path='results',save_name='oe_detect'):
        self.args = args
        self.save_path = save_path
        self.aucs=[]
        self.save_name = save_name

    def add_result(self,result):
        self.aucs.append(result)

    def cal_avg_std_auc(self):
        avg_auc = np.mean(self.aucs)
        std_auc = np.std(self.aucs)
        return avg_auc, std_auc 
    
    # def save_result(self,args):
    #     if args.exp_type == 'ood':
    #         if args.OE:
    #             filename = f'{self.save_path}/{args.DS_pair}-{args.DS_oe}.csv'
    #         else:
    #             filename = f'{self.save_path}/{args.DS_pair}.csv'
    #     else:
    #         filename = f'{self.save_path}/{args.dataset}.csv'

    #     print(f"Saving results to {filename}")
    #     with open(f"{filename}", 'a+') as write_obj:
    #         write_obj.write(f"EXP TYPE:{args.exp_type}\n")
    #         write_obj.write(f'IND Test Score: {r.mean():.2f} ± {r.std():.2f}\n')
    #         write_obj.write(f'\n')

    def save_result(self,args):
        if args.exp_type == 'ood':
            if args.OE:
                filename = f'{self.save_path}/{args.DS_pair}-{args.DS_oe}.csv'
            else:
                filename = f'{self.save_path}/{args.DS_pair}.csv'
        else:
            filename = f'{self.save_path}/{args.dataset}.csv'

        print(f"Saving results to {filename}")
        with open(f"{filename}", 'a+') as write_obj:
            write_obj.write(f"EXP TYPE:{args.exp_type}\n")
            write_obj.write(f'IND Test Score: {r.mean():.2f} ± {r.std():.2f}\n')
            write_obj.write(f'\n')