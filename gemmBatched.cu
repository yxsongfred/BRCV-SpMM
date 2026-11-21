/*
    An implementation of batched matrix-matrix multiplication (GEMM) accelerated by cuBLAS
    Ideas inspired by NVIDIA(R) CUDALibrarySamples
    Reference
        [1] cuBLAS Level-3 APIs - cublas<t>gemmBatched
        https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Level-3/gemmBatched/cublas_gemmBatched_example.cu
    Re-coded and annotated by Yuxuan SONG
    Last modified time: Apr 27, 2024
*/
#include <iostream>
#include <vector>
#include "utils.h"
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define ITER_MAX 100

using namespace std;

int main(int argc, char *argv[])
{
    /*
        """
            STEP ZERO
            DATA PREPARATION (EXCLUDED IN TIME STATISTICS)
        """
    */
    // Obtain the directory of dataset folder according to the input parameter
    string folderdir, filepfx;
    datasetMap(argc, argv, folderdir);
    /*
        Obtain dimension parameters:
            idx[i] = M^{(i)}
            idx[i + 1] = N^{(i)}
            idx[i + 2] = K^{(i)}
            idx[i + 3] = i
        A^{(i)}\in\mathbb{R}^{M^{(i)}\times K^{(i)}}
        B^{(i)}\in\mathbb{R}^{K^{(i)}\times N^{(i)}}
        C^{(i)}\in\mathbb{R}^{M^{(i)}\times N^{(i)}}
    */
    vector<int> idx;
    readcsv(folderdir + "indices.csv", idx);
    const int batch_count = idx.size() / 4;
    // CSR-format data loader
    vector<int> vec_rpt, vec_cid;
    vector<float> vec_val;
    // Dense-format data loader
    vector<float> tA, tB, tC, ttA, ttB, ttC;
    vector<vector<float>> h_A, h_B, h_C;
    /*
        All instances have the same dimensions (M, N, K)
    */
    filepfx = folderdir + to_string(idx[0]) + "_" + to_string(idx[1]) + "_" + to_string(idx[2]) + "_";
    for (int i = 0; i < idx.size(); i += 4)
    {
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "cid.csv", vec_cid);
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "rpt.csv", vec_rpt);
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "val.csv", vec_val);
        csr2dense(vec_rpt, vec_cid, vec_val, idx[2], tA);
        vec_cid.clear();
        vec_rpt.clear();
        vec_val.clear();
        // VITAL!!!! Dense matrices must be transposed!!!
        /*
            idx[0] = M, idx[1] = N, idx[2] = K
            dims of A : M * K;   transposed A : K * M
            dims of B : K * N;   transposed B : N * K
            dims of C : M * N;   transposed C : N * M
        */
        mtxTranspose(tA, ttA, idx[0]);
        tA.clear();
        h_A.push_back(ttA);
        ttA.clear();
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "B.csv", tB);
        mtxTranspose(tB, ttB, idx[2]);
        tB.clear();
        h_B.push_back(ttB);
        ttB.clear();
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "C.csv", tC);
        mtxTranspose(tC, ttC, idx[0]);
        tC.clear();
        h_C.push_back(ttC);
        ttC.clear();
    }
    // --------------------------------------------------------------------------
    // Host problem definition
    const int m = idx[0];
    const int n = idx[1];
    const int k = idx[2];
    idx.clear();
    // Number of Floating-point Operations
    const long long FLOP = batch_count * static_cast<long long>(m) * static_cast<long long>(k) * static_cast<long long>(n) * 2;
    /*
        ldX means the leading dimension of matrix X
        ldX = #cols in X iff X is row-major stored
        ldX = #rows in X iff X is column-major stored
    */
    const int lda = m;
    const int ldb = k;
    const int ldc = m;
    const float alpha = 1.0;
    const float beta = 0.0;
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            STEP ONE
            INITIALIZATION FOR CUBLAS
        """
    */
    auto init_start = chrono::high_resolution_clock::now();
    // --------------------------------------------------------------------------
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    cublasCreate(&cublasH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(cublasH, stream);
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    // --------------------------------------------------------------------------
    auto init_stop = chrono::high_resolution_clock::now();
    float init_time = chrono::duration<float, std::milli>(init_stop - init_start).count();
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            STEP TWO
            CPU->GPU COMMUNICATIONS
        """
    */
    // CUDA memory allocation
    auto mcph2d_start = chrono::high_resolution_clock::now();
    // --------------------------------------------------------------------------
    float **d_A_array = nullptr;
    float **d_B_array = nullptr;
    float **d_C_array = nullptr;
    vector<float *> d_A(batch_count, nullptr);
    vector<float *> d_B(batch_count, nullptr);
    vector<float *> d_C(batch_count, nullptr);
    for (int i = 0; i < batch_count; i++)
    {
        cudaMalloc(reinterpret_cast<void **>(&d_A[i]), sizeof(float) * h_A[i].size());
        cudaMalloc(reinterpret_cast<void **>(&d_B[i]), sizeof(float) * h_B[i].size());
        cudaMalloc(reinterpret_cast<void **>(&d_C[i]), sizeof(float) * h_C[i].size());
    }
    cudaMalloc(reinterpret_cast<void **>(&d_A_array), sizeof(float *) * batch_count);
    cudaMalloc(reinterpret_cast<void **>(&d_B_array), sizeof(float *) * batch_count);
    cudaMalloc(reinterpret_cast<void **>(&d_C_array), sizeof(float *) * batch_count);
    // Memory copy from CPU to GPU
    for (int i = 0; i < batch_count; i++)
    {
        cudaMemcpyAsync(d_A[i], h_A[i].data(), sizeof(float) * h_A[i].size(),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_B[i], h_B[i].data(), sizeof(float) * h_B[i].size(),
                        cudaMemcpyHostToDevice, stream);
    }
    cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(float *) * batch_count,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(float *) * batch_count,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(float *) * batch_count,
                    cudaMemcpyHostToDevice, stream);
    auto mcph2d_stop = chrono::high_resolution_clock::now();
    float mcph2d_time = chrono::duration<float, std::milli>(mcph2d_stop - mcph2d_start).count();
    // --------------------------------------------------------------------------
    // Clear or free useless vectors on CPU
    h_A.clear();
    h_B.clear();
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            STEP THREE
            GPU KERNEL COMPLEMENTATION
            BATCHED GEMM
        """
    */
    // Warm-up
    for (int it = 0; it < 10; it++)
        cublasSgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda,
                           d_B_array, ldb, &beta, d_C_array, ldc, batch_count);
    // ---------------------------------------------------------------------------
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventRecord(kernel_start, 0);
    // --------------------------------------------------------------------------
    // Average on several implementations
    for (int it = 0; it < ITER_MAX; it++)
        cublasSgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda,
                           d_B_array, ldb, &beta, d_C_array, ldc, batch_count);
    // --------------------------------------------------------------------------
    cudaEventRecord(kernel_stop, 0);
    cudaEventSynchronize(kernel_stop);
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            STEP FOUR
            GPU->CPU COMMUNICATIONS
        """
    */
    auto mcpd2h_start = chrono::high_resolution_clock::now();
    // --------------------------------------------------------------------------
    vector<vector<float>> res(batch_count, vector<float>(m * n));
    for (int i = 0; i < batch_count; i++)
        cudaMemcpyAsync(res[i].data(), d_C[i], sizeof(float) * res[i].size(),
                        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    // --------------------------------------------------------------------------
    auto mcpd2h_stop = chrono::high_resolution_clock::now();
    float mcpd2h_time = chrono::duration<float, std::milli>(mcpd2h_stop - mcpd2h_start).count();
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            STEP FIVE
            FREE MEMORY
        """
    */
    auto free_start = chrono::high_resolution_clock::now();
    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_C_array);
    for (int i = 0; i < batch_count; i++)
    {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_C_array);
    // --------------------------------------------------------------------------
    auto free_stop = chrono::high_resolution_clock::now();
    float free_time = chrono::duration<float, std::milli>(free_stop - free_start).count();
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            STEP SIX
            DESTRUCTION OF CUBLAS
        """
    */
    auto des_start = chrono::high_resolution_clock::now();
    // --------------------------------------------------------------------------
    cublasDestroy(cublasH);
    cudaStreamDestroy(stream);
    cudaDeviceReset();
    // --------------------------------------------------------------------------
    auto des_stop = chrono::high_resolution_clock::now();
    float des_time = chrono::duration<float, std::milli>(des_stop - des_start).count();
    // --------------------------------------------------------------------------
    // Verify the SpMM results
    verify(res, h_C);
    // --------------------------------------------------------------------------
    // Free space
    h_C.clear();
    res.clear();
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            REPORT OF RESULTS
        """
    */
    // Compute GFLOPS (Giga Floating-point Operations Per Second)
    float GFLOPS = FLOP * 1.0 / (kernel_time / ITER_MAX / 10e3) / 10e9;
    cout << "Batched GEMM" << endl
         << "GFLOPS:\t"
         << fixed << setprecision(6)
         << GFLOPS << endl
         << "Time statistics (ms):" << endl
         << "cuBLAS handle & stream initialization\t" << init_time << endl
         << "cudaMalloc & cudaMemcpyHostToDevice\t" << mcph2d_time << endl
         << "CUDA kernel function\t" << kernel_time / ITER_MAX << endl
         << "cudaMemcpyDevicetoHost\t" << mcpd2h_time << endl
         << "cudaFree\t" << free_time << endl
         << "cuBLAS handle & stream destruction\t" << des_time << endl;
    return 0;
}