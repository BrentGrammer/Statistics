# Installing PyTorch on Windows

- Check Cuda version in Powershell: `nvidia-smi.exe`
- Install PyTorch with command from the [Pytorch website](https://pytorch.org/)
  - this worked: `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
- Verify Installation by running this python code:

```python
import torch
print(f'PyTorch version: {torch.__version__}')
print('*'*10)
print(f'_CUDA version: ')
!nvcc --version
print('*'*10)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')
# should get no errors from the output
```
