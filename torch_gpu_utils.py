import torch
import subprocess

def get_devices(max_devices=1):
    """
    ARGS: max_devices | int | number of devices required
    RETURNS: devices | torch.device or [torch.device,...] | cpu or gpus with most free memory 
    """
    
    if torch.cuda.is_available():
        gpu_list = get_gpu_memory_map()
        gpu_list.sort(key=lambda tup: tup[1], reverse=True)
        gpu_indices = [i[0] for i in gpu_list[:max_devices]]
        devices = [torch.device(i) for i in gpu_indices]
        if len(devices)==1:
            devices=devices[0]
    else:
        devices = torch.device("cpu")
        gpu_indices = None
        
        
    return devices, gpu_indices

def get_gpu_memory_map():
    """Get the free amount of memory on gpus.
    RETURNS: gpu_memory_map | list | [[device id, free mem in MB],...]
    """
    
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = list(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map
