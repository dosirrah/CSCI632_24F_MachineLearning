/**
 * CUDA-example for adding two vectors using parallel processing
 * within a GPU.   The language is based on C, but with some
 * extensions to support parallelism using SIMT = Single Instruction
 * Multiple Threads.
 */

__global__ void add(int *a, int *b, int *c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {  // Ensure we do not perform out-of-bounds access
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n = 10000;  // Example: total number of elements in each vector
    int *a, *b, *c; // Device pointers
    int *a_host, *b_host, *c_host; // Host pointers

    // Allocate memory on host
    a_host = (int*)malloc(n * sizeof(int));
    b_host = (int*)malloc(n * sizeof(int));
    c_host = (int*)malloc(n * sizeof(int));

    // Initialize vectors on host
    for(int i = 0; i < n; i++) {
        a_host[i] = i;
        b_host[i] = i;
    }

    // Allocate memory on device
    cudaMalloc(&a, n * sizeof(int));
    cudaMalloc(&b, n * sizeof(int));
    cudaMalloc(&c, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(a, a_host, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b, b_host, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launching kernel
    int blockSize = 256;  // Number of threads in each block
    int numBlocks = (n + blockSize - 1) / blockSize;  // Number of blocks
    add<<<numBlocks, blockSize>>>(a, b, c, n);

    // Copy result back to host
    cudaMemcpy(c_host, c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(a); cudaFree(b); cudaFree(c);
    free(a_host); free(b_host); free(c_host);

    return 0;
}

