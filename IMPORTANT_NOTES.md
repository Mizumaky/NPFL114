Python > 3.8 and PATH / DLL libraries problem with UWP Python on Windows
[Solution](https://stackoverflow.com/questions/62971631/tensorflow-cannot-find-dll-but-dll-directory-is-in-path)
add these lines to C:\envs\dl\Lib\site-packages\tensorflow\python\__init__.py:
```python
import os
os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.3/bin")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/include")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/extras/CUPTI/lib64")
```