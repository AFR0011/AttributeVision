import torch
print(torch.__version__)  # e.g., 2.5.0+cu124
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)  # e.g., 12.4
print(torch.cuda.device_count())  # Number of GPUs (e.g., 1)
print(torch.cuda.get_device_name(0))  # GPU name (e.g., GeForce RTX 2060)