import re
from datetime import datetime
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#import horovod.torch as hvd
import os

class Communication(object):

    def __init__(self, lib="pytorch"):
        print(lib)
        if lib == "pytorch": 
            self.lib = "pytorch"
            self.handler = dist
            self.handler.init_process_group(backend='nccl', init_method="tcp://" + self.get_master_ip() + ":" + self.get_master_port(), rank=self.get_rank(), world_size=self.get_world())
            torch.cuda.set_device(self.get_local_rank())
        elif lib == "horovod":
            self.lib = "horovod"
            print("self.handler = hvd")
            self.handler = hvd
            print("self.handler.init()")
            self.handler.init()
            if not self.handler.nccl_built():
                raise Exception("NCCL was not compiled in Horovod!")
        else:
            raise ValueError("Only Pytorch DDP lib and Horovod are currently supported.")
    
    def sync(self):

        if self.lib == "pytorch":
            self.handler.barrier()
        else: # Horovod
            print("self.handler.broadcast_object(0, root_rank=0)")
            self.handler.broadcast_object(0, root_rank=0)

    def broadcast_model(self, model):

        if self.lib == "pytorch":
            model.cuda(torch.cuda.current_device())
            print("before DDP")
            model = DDP(model, device_ids=[self.get_local_rank()])
            print("after DDP")
        else: # Horovod
            print("self.handler.broadcast_parameters(model.state_dict(), root_rank=0)")
            self.handler.broadcast_parameters(model.state_dict(), root_rank=0)
        
        return model
    
    def broadcast_optimizer(self, model, optimizer):
        if self.lib == "horovod":
            print("optimizer = self.handler.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())")
            optimizer = self.handler.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        
        return optimizer

    def all_reduce(self, data):

        if self.lib == "pytorch":
            self.handler.all_reduce(data, op=dist.ReduceOp.SUM)
            data /= float(self.get_world())
        else: # Horovod
            print("data = self.handler.allreduce(data)")
            data = self.handler.allreduce(data)

        return data
        
        
    def get_rank(self):

        rank = 0
        if os.environ.get('OMPI_COMM_WORLD_RANK') is not None: 
            rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
        elif os.environ.get('RANK') is not None: 
            print("rank = int(os.environ.get('RANK'))")
            rank = int(os.environ.get('RANK'))
        
        return rank

    def get_local_rank(self):

        rank = 0

        if os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') is not None: 
            rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK'))
        elif os.environ.get('LOCAL_RANK') is not None: 
            print("rank = int(os.environ.get('LOCAL_RANK'))")
            rank = int(os.environ.get('LOCAL_RANK'))
        

        return rank
    
    def get_world(self):

        world = 1
        if os.environ.get('OMPI_COMM_WORLD_SIZE') is not None:
            print("world = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))")
            world = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
        elif os.environ.get('WORLD_SIZE') is not None:
            print("world = int(os.environ.get('WORLD_SIZE'))")
            world = int(os.environ.get('WORLD_SIZE'))

        return world


    def get_master_ip(self):
        if os.environ.get('MASTER_ADDR') is not None: 
            ip = os.environ.get('MASTER_ADDR')
            print("master node ip is " + ip)
            return ip
        else:
            raise ValueError("did not find master node ip")

    def get_master_port(self):
        if os.environ.get('MASTER_PORT') is not None: 
            port = os.environ.get('MASTER_PORT')
            print("master node port is " + port)
            return port
        else:
            raise ValueError("did not find master node port")

    
