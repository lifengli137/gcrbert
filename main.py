import time
import torch
import os
import numpy as np
from importlib import import_module
import argparse
import utils
import train
from communication import Communication
from pytorch_pretrained_bert import BertModel, BertTokenizer

parser = argparse.ArgumentParser(description='GCR-Bert-Text-Classification.')
parser.add_argument('--model', type=str, default='bert', help='Choose a model: bert')
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--dataset", type=str, default="THUCNews")
parser.add_argument("--data_path", type=str, default=".")
parser.add_argument("--checkpoint_path", type=str, default=".")
parser.add_argument("--job_name", type=str, default="gcr_test_job")
parser.add_argument("--model_path", type=str, default=os.getenv('AMLT_DATA_DIR', 'data'))
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--learning_rate", type=float, default=1e-7)
parser.add_argument("--warmup", type=float, default=0.05)
parser.add_argument("--lib", type=str, default="horovod")

args = parser.parse_args()

if __name__ == '__main__':

    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(args)
    print("1")
    config.comm = Communication(lib=args.lib)
    print("2")

#    if config.comm.get_local_rank() == 0:
    BertModel.from_pretrained('bert-base-chinese')
    BertTokenizer.from_pretrained('bert-base-chinese') 
#    config.comm.sync()

    print("3")
    model = x.Model(config)
    print("4")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    torch.cuda.set_device(config.comm.get_local_rank())
    print("5")
#    try:
#        config.checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
#        config.continue_training = True
#        config.comm.sync()
#    except (FileNotFoundError, TypeError):
    model = config.comm.broadcast_model(model)
    print("New training...")

    if config.continue_training:
        model.load_state_dict(config.checkpoint['model_state_dict'])
        config.start_epoch = config.checkpoint['epoch']
        config.start_loss = config.checkpoint['loss']
        print("Contrinue training...")

    start_time = time.time()
    print('Loading dataset')

    train_data, dev_data, test_data = utils.build_dataset(config)
    train_iter = utils.build_dataloader(train_data, config, training=True)
    dev_iter = utils.build_dataloader(dev_data, config, training=False)
    test_iter = utils.build_dataloader(test_data, config, training=False)

    time_dif = utils.get_time_dif(start_time)
    print("Prepare data time: ", time_dif)

    model = model.to(config.device)

    train.train(config, model, train_iter, dev_iter, test_iter)
