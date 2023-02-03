from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import time
from datetime import timedelta
import pickle as pkl
import os
from pytorch_pretrained_bert import BertTokenizer

def get_world_rank():                                             
                                                                
    rank = 0
    if os.environ.get('OMPI_COMM_WORLD_RANK') is not None: 
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
    elif os.environ.get('RANK') is not None: 
        rank = int(os.environ.get('RANK'))                         
                                                                
    return rank                                                 
                                                                
def get_local_rank():                                       
                                                                
    rank = 0

    if os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') is not None: 
        rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK'))
    elif os.environ.get('LOCAL_RANK') is not None: 
        rank = int(os.environ.get('LOCAL_RANK'))                                                         
                                                                
    return rank                                                 
                                                                
def get_world():                                            
                                                                
    world = 1
    if os.environ.get('OMPI_COMM_WORLD_SIZE') is not None:
        world = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
    elif os.environ.get('WORLD_SIZE') is not None:
        world = int(os.environ.get('WORLD_SIZE'))               
                                                                
    return world                                            



def all_reduce(data):

    torch.distributed.all_reduce(data, op=torch.distributed.ReduceOp.SUM)
    data /= float(get_world())
 
    return data


PAD, CLS = '[PAD]', '[CLS]'


def load_dataset(data_path, config):
    """
    return:  4 lists: ids, label, ids_len, mask
    """

    if not config.tokenizer:
        config.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    contents = []
    with open(data_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):

            line = line.strip()
            if not line:
                continue

            content, label = line.split('\t')
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            pad_size = config.pad_size
            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids = token_ids + ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size

            contents.append((token_ids, int(label), seq_len, mask))

    return contents

def tensorize_dataset(dataset):

    token_ids_tensor = torch.tensor([c[0] for c in dataset], dtype=torch.long)
    label_tensor = torch.tensor([c[1] for c in dataset], dtype=torch.long)
    seq_len_tensor = torch.tensor([c[2] for c in dataset], dtype=torch.long)
    mask_tensor = torch.tensor([c[3] for c in dataset], dtype=torch.long)
    tensor_dataset = TensorDataset(token_ids_tensor, label_tensor, seq_len_tensor, mask_tensor)

    return tensor_dataset


def build_dataset(config):
    """
    return: train, dev, test
    """

    if os.path.exists(config.datasetpkl):
        dataset = pkl.load(open(config.datasetpkl, 'rb'))
        train = dataset['train']
        dev = dataset['dev']
        test = dataset['test']
    else:
        train = load_dataset(config.train_path, config)
        dev = load_dataset(config.dev_path, config)
        test = load_dataset(config.test_path, config)

        train = tensorize_dataset(train)
        dev = tensorize_dataset(dev)
        test = tensorize_dataset(test)
        dataset = {}
        dataset['train'] = train
        dataset['dev'] = dev
        dataset['test'] = test
        pkl.dump(dataset, open(config.datasetpkl, 'wb'))

    return train, dev, test


def get_time_dif(start_time):
    """
    Get the time difference
    """

    end_time = time.time()
    time_dif = end_time - start_time

    return timedelta(seconds=int(round(time_dif)))


def build_dataloader(data, config, training=True):
    if training is True:
        sampler = DistributedSampler(data, num_replicas=get_world(), rank=get_world_rank())
    else:
        sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=config.batch_size)
    return dataloader