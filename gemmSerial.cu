/*
    A serial implementation of batched matrix-matrix multiplication (GEMM) accelerated by cuBLAS
    Ideas inspired by NVIDIA(R) CUDALibrarySamples
    Reference
        [1] cuBLAS Level-3 APIs - cublas<t>gemm
        https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Level-3/gemm/cublas_gemm_example.cu
    Re-coded and annotated by Yuxuan SONG
    Last modified time: Apr 27, 2024
*/
#include <iostream>
#include <vector>
#include "utils.h"
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

int main(int argc, char *argv[])
{
    string folderdir, filepfx;
    // Obtain the directory of dataset folder according to the input parameter
    datasetMap(argc, argv, folderdir);
    vector<int> idx;
    readcsv(folderdir + "indices.csv", idx);
    // CSR-format data loader
    vector<int> vec_rpt, vec_cid;
    vector<float> vec_val;
    // Dense-format data loader
    vector<float> tA, tB, tC, h_A, h_B, h_C;
    float init_time = 0, mcph2d_time = 0, kernel_time = 0, mcpd2h_time = 0, free_time = 0, des_time = 0;
    // Number of Floating-point Operations
    long long FLOP = 0;
    const float alpha = 1.0;
    const float beta = 0.0;
    for (int i = 0; i < idx.size(); i += 4)
    {
        /*
            """
                STEP ZERO
                DATA PREPARATION (EXCLUDED IN TIME STATISTICS)
            """
        */
        /*
            Obtain dimension parameters:
                idx[i] = M^{(i + 1)}
                idx[i + 1] = N^{(i + 1)}
                idx[i + 2] = K^{(i + 1)}
                idx[i + 3] = i
            A^{(i + 1)}\in\mathbb{R}^{M^{(i + 1)}\times K^{(i + 1)}}
            B^{(i + 1)}\in\mathbb{R}^{K^{(i + 1)}\times N^{(i + 1)}}
            C^{(i + 1)}\in\mathbb{R}^{M^{(i + 1)}\times N^{(i + 1)}}
        */
        filepfx = folderdir + to_string(idx[i]) + "_" + to_string(idx[i + 1]) + "_" + to_string(idx[i + 2]) + "_";
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "cid.csv", vec_cid);
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "rpt.csv", vec_rpt);
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "val.csv", vec_val);
        csr2dense(vec_rpt, vec_cid, vec_val, idx[i + 2], tA);
        // VITAL!!!! Dense matrices must be transposed!!!
        /*
            idx[i] = M, idx[i + 1] = N, idx[i + 2] = K
            dims of A : M * K;   transposed A : K * M
            dims of B : K * N;   transposed B : N * K
            dims of C : M * N;   transposed C : N * M
        */
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
        mtxTranspose(tA, h_A, idx[i]);
        tA.clear();
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "B.csv", tB);
        mtxTranspose(tB, h_B, idx[2]);
        tB.clear();
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "C.csv", tC);
        mtxTranspose(tC, h_C, idx[0]);
        tC.clear();
        // --------------------------------------------------------------------------
        // Host problem definition
        int m = idx[i];
        int n = idx[i + 1];
        int k = idx[i + 2];
        int lda = m;
        int ldb = k;
        int ldc = m;
        // --------------------------------------------------------------------------
        FLOP += m * k * n * 2;
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
        init_time += chrono::duration<float, std::milli>(init_stop - init_start).count();
        // --------------------------------------------------------------------------
        // --------------------------------------------------------------------------
        /*
            """
                STEP TWO
                CPU->GPU COMMUNICATIONS
            """
        */
        auto mcph2d_start = chrono::high_resolution_clock::now();
        float *d_A = nullptr;
        float *d_B = nullptr;
        float *d_C = nullptr;
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * h_A.size());
        cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * h_B.size());
        cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(float) * h_C.size());
        cudaMemcpyAsync(d_A, h_A.data(), sizeof(float) * h_A.size(),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_B, h_B.data(), sizeof(float) * h_B.size(),
                        cudaMemcpyHostToDevice, stream);
        auto mcph2d_stop = chrono::high_resolution_clock::now();
        mcph2d_time += chrono::duration<float, std::milli>(mcph2d_stop - mcph2d_start).count();
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
                SINGLE GEMM
            """
        */
        cudaEvent_t kernel_start, kernel_stop;
        cudaEventCreate(&kernel_start);
        cudaEventCreate(&kernel_stop);
        cudaEventRecord(kernel_start, 0);
        cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
        cudaEventRecord(kernel_stop, 0);
        cudaEventSynchronize(kernel_stop);
        float single_kernel_time;
        cudaEventElapsedTime(&single_kernel_time, kernel_start, kernel_stop);
        kernel_time += single_kernel_time;
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
        float *res = (float *)malloc(h_C.size() * sizeof(float));
        cudaMemcpyAsync(res, d_C, sizeof(float) * h_C.size(), cudaMemcpyDeviceToHost,
                        stream);
        cudaStreamSynchronize(stream);
        // --------------------------------------------------------------------------
        auto mcpd2h_stop = chrono::high_resolution_clock::now();
        mcpd2h_time += chrono::duration<float, std::milli>(mcpd2h_stop - mcpd2h_start).count();
        // --------------------------------------------------------------------------
        // --------------------------------------------------------------------------
        /*
            """
                STEP FIVE
                FREE MEMORY
            """
        */
        auto free_start = chrono::high_resolution_clock::now();
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        // --------------------------------------------------------------------------
        auto free_stop = chrono::high_resolution_clock::now();
        free_time += chrono::duration<float, std::milli>(free_stop - free_start).count();
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
        des_time += chrono::duration<float, std::milli>(des_stop - des_start).count();
        // --------------------------------------------------------------------------
        // Verify the SpMM results
        verify(res, h_C);
        // --------------------------------------------------------------------------
        // Free space
        free(res);
        h_C.clear();
    }
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            REPORT OF RESULTS
        """
    */
    // Compute GFLOPS (Giga Floating-point Operations Per Second)
    float GFLOPS = FLOP * 1.0 / (kernel_time / 10e3) / 10e9;
    cout << "Serial GEMM" << endl
         << "GFLOPS:\t"
         << fixed << setprecision(6)
         << GFLOPS << endl
         << "Time statistics (ms):" << endl
         << "cuBLAS handle & stream initialization\t" << init_time << endl
         << "cudaMalloc & cudaMemcpyHostToDevice\t" << mcph2d_time << endl
         << "CUDA kernel function\t" << kernel_time << endl
         << "cudaMemcpyDevicetoHost\t" << mcpd2h_time << endl
         << "cudaFree\t" << free_time << endl
         << "cuBLAS handle & stream destruction\t" << des_time << endl;
    return 0;
}