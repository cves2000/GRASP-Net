from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CppExtension,CUDAExtension

# 使用 setuptools 的 setup 函数来设置包的信息
# setup(name='my_lib',
#      ext_modules=[CppExtension('my_lib', ['src/my_lib.cpp'])],  # 使用 CppExtension 来编译 C++ 扩展模块
#      cmdclass={'build_ext': BuildExtension})  # 使用 PyTorch 提供的 BuildExtension 来构建扩展模块

# 设置包的名称为 'my_lib_cuda'
# 扩展模块为 CUDAExtension，包含两个源文件 'src/my_lib_cuda.cpp' 和 'src/nnd_cuda.cu'
# cmdclass 用于指定 build_ext 命令的处理器，这里使用 PyTorch 提供的 BuildExtension
setup(name='my_lib_cuda',
      ext_modules=[CUDAExtension('my_lib_cuda',['src/my_lib_cuda.cpp', 'src/nnd_cuda.cu']
              )],
      cmdclass={'build_ext': BuildExtension}
      )

# 如果这个脚本被直接运行，而不是被导入，那么就会执行 ffi.build()
#if __name__ == '__main__':
#    ffi.build()

# 这段代码的主要功能是编译和安装名为 ‘my_lib_cuda’ 的 Python 扩展模块。这个模块包含两个源文件：‘src/my_lib_cuda.cpp’ 和 ‘src/nnd_cuda.cu’。
# 这段代码使用了 PyTorch 的 torch.utils.cpp_extension 模块，该模块提供了一种方便的方式来编译和加载 C++ 或 CUDA 扩展。
# 这对于需要在 Python 中使用 C++ 或 CUDA 代码的情况非常有用，例如，当你需要在 PyTorch 中实现自定义的操作时。
# 注意，这段代码需要在支持 CUDA 的环境中运行，否则会报错。如果你想在没有 CUDA 的环境中运行这段代码，
# 你可以注释掉 CUDAExtension 相关的代码，然后取消注释 CppExtension 相关的代码。这样，你就可以编译和安装名为 ‘my_lib’ 的 C++ 扩展模块了。
