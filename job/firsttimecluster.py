import torch
print("hi")
print("You have been assigned an " + str(torch.cuda.get_device_name(0)) +" ! Don't forget to have fun while you explore. :-)")
print("Akansh"+ str(torch.cuda.is_available()))
print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
import timm
import subprocess
subprocess.run(["nvidia-smi"])
