#include <iostream>
#include <cuda_runtime.h>

void printDeviceProperties()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << "\n"
                  << "  Compute capability: " << prop.major << "." << prop.minor << "\n"
                  << "  Global memory: " << (prop.totalGlobalMem >> 20) << " MB\n"
                  << "  Shared memory per block: " << prop.sharedMemPerBlock << " bytes\n"
                  << "  Registers per block: " << prop.regsPerBlock << "\n"
                  << "  Warp size: " << prop.warpSize << "\n"
                  << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n"
                  << "  Max thread dimensions: [" << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << "]\n"
                  << "  Max grid dimensions: [" << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << "]\n"
                  << std::endl;
    }
}

int main()
{
    printDeviceProperties();
    return 0;
}