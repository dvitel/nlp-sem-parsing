import torch 
import psutil
import platform
import multiprocessing

dev_props = torch.cuda.get_device_properties("cuda:0")
print("GPU")
print(dev_props)

mem = psutil.virtual_memory()
print("Memory")
print(mem)

cpu = platform.processor()
print("Processor")
print(cpu)
print("Num processors: ", multiprocessing.cpu_count())