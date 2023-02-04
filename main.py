import time                                                   
import torch                                                  
import os                                                     
import numpy as np                                            
from importlib import import_module                           
import argparse      
import sys                                                   
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.nn.functional as F                           
from sklearn import metrics                               
from shutil import copyfile                               
from pytorch_pretrained_bert.optimization import BertAdam 
import argparse                                     
from pytorch_pretrained_bert import BertModel, BertTokenizer  
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta
import pickle as pkl
import utils
import train

parser = argparse.ArgumentParser(description='GCR-Bert-Text-Classification.')                  
parser.add_argument('--model', type=str, default='bert', help='Choose a model: bert')          
parser.add_argument("--seed", type=int, default=1234)                                          
parser.add_argument("--batch_size", type=int, default=128)                                     
parser.add_argument("--epochs", type=int, default=1)                                           
parser.add_argument("--dataset", type=str, default="THUCNews")                                 
parser.add_argument("--data_path", type=str, default="/tmp/")                                      
parser.add_argument("--checkpoint_path", type=str, default="/tmp/")                                
parser.add_argument("--job_name", type=str, default="gcr_test_job")                            
parser.add_argument("--model_path", type=str, default="/tmp/")  
parser.add_argument("--weight_decay", type=float, default=0.01)                                
parser.add_argument("--learning_rate", type=float, default=1e-7)                               
parser.add_argument("--warmup", type=float, default=0.05)                                      
                                                                                               
args = parser.parse_args()                                       


if __name__ == '__main__':                                          

    seed = 1                                    
    model_name = args.model                                         
    x = import_module('models.' + model_name)                       
    config = x.Config(args)                                         

    np.random.seed(seed)                                       
    torch.manual_seed(seed)                                    
    torch.cuda.manual_seed_all(seed)                           
    torch.backends.cudnn.deterministic = True      

    ip = "localhost"
    if os.environ.get('MASTER_ADDR') is not None:      
        ip = os.environ.get('MASTER_ADDR')             

    port = "23456"
    if os.environ.get('MASTER_PORT') is not None:        
        port = os.environ.get('MASTER_PORT')             


    torch.distributed.init_process_group(backend='nccl', init_method="tcp://" + ip + ":" + port, rank=utils.get_world_rank(), world_size=utils.get_world()) 

    if utils.get_local_rank() == 0: # Only rank 0 per node download the model here
        BertModel.from_pretrained('bert-base-chinese')                  
        BertTokenizer.from_pretrained('bert-base-chinese')      
    torch.distributed.barrier()        
                                                                        
    model = x.Model(config)                                         
                                                                    
    

    torch.cuda.set_device(utils.get_local_rank())                                                                                                                      
    model.cuda(torch.cuda.current_device())               
    model = DDP(model, device_ids=[utils.get_local_rank()])                                  
                                                                    
                     
                                                                    
    start_time = time.time()                                        
    print('Loading dataset')                                        
                                                                    
    train_data, dev_data, test_data = utils.build_dataset(config)         
    train_iter = utils.build_dataloader(train_data, config, training=True)
    dev_iter = utils.build_dataloader(dev_data, config, training=False)   
    test_iter = utils.build_dataloader(test_data, config, training=False) 
                                                                    
    time_dif = utils.get_time_dif(start_time)                       
    print("Prepare data time: ", time_dif)                          
                                                                    
    model = model.to(torch.cuda.current_device())                                 
                                                                    

    train.train(config, model, train_iter, dev_iter, test_iter)

