/*
    An implementation of batched CSR-format sparse matrix-matrix multiplication accelerated by cuSPARSE
    Ideas inspired by NVIDIA(R) CUDALibrarySamples
    Reference
        [1] cuSPARSE Generic APIs - Batched cusparseSpMM CSR
        https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spmm_csr_batched/spmm_csr_batched_example.c
    Re-coded and annotated by Yuxuan SONG
    Last modified time: Apr 27, 2024
*/
#include <iostream>
#include <vector>
#include "utils.h"
#include <chrono>
#include <cuda_runtime.h>
#include <cusparse.h>

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
    folderdir = "./data/eqMNKrpt/M64_N8_K64/1/";
    // datasetMap(argc, argv, folderdir);
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
    int num_batches = idx.size() / 4;
    // CSR-format data loader
    vector<int> h_rpt, h_cid;
    vector<float> h_val;
    // Dense-format data loader
    vector<float> tB, ttB, h_B, tC, ttC, h_C;
    /*
        All instances have the same dimensions (M, N, K)
        All instances of matrices A's have the same rpt arrays in their CSR-format storations
    */
    for (int i = 0; i < idx.size(); i += 4)
    {
        if (i == 0)
        {
            filepfx = folderdir + to_string(idx[0]) + "_" + to_string(idx[1]) + "_" + to_string(idx[2]) + "_";
            readcsv(filepfx + to_string(idx[3]) + "_" + "rpt.csv", h_rpt);
        }
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "cid.csv", h_cid);
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "val.csv", h_val);
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "B.csv", tB);
        readcsv(filepfx + to_string(idx[i + 3]) + "_" + "C.csv", tC);
        /*
            VITAL!!!! Dense matrices must be transposed!!!
            Convert row-major storation to column-major storation
        */
        /*
            idx[0] = M, idx[1] = N, idx[2] = K
            dims of B : K * N;   transposed B : N * K
            dims of C : M * N;   transposed C : N * M
        */
        mtxTranspose(tB, ttB, idx[2]);
        tB.clear();
        mtxTranspose(tC, ttC, idx[0]);
        tC.clear();
        vecAppend(ttB, h_B);
        ttB.clear();
        vecAppend(ttC, h_C);
        ttC.clear();
    }
    // --------------------------------------------------------------------------
    // Host problem definition
    const int A_num_rows = idx[0], A_num_cols = idx[2], A_nnz = h_val.size() / num_batches;
    const int B_num_rows = A_num_cols, B_num_cols = idx[1];
    const int ldb = B_num_rows, ldc = A_num_rows;
    const int B_size = ldb * B_num_cols, C_size = ldc * B_num_cols;
    // Number of Floating-point Operations
    const long long FLOP = static_cast<long long>(h_val.size()) * static_cast<long long>(B_num_cols) * 2;
    idx.clear();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            STEP ONE
            CPU->GPU COMMUNICATIONS
        """
    */
    auto mcph2d_start = chrono::high_resolution_clock::now();
    // --------------------------------------------------------------------------
    int *d_rpt, *d_cid;
    float *d_val, *d_B, *d_C;
    // CUDA memory allocation
    cudaMalloc((void **)&d_rpt, (A_num_rows + 1) * sizeof(int));
    cudaMalloc((void **)&d_cid, A_nnz * num_batches * sizeof(int));
    cudaMalloc((void **)&d_val, A_nnz * num_batches * sizeof(float));
    cudaMalloc((void **)&d_B, B_size * num_batches * sizeof(float));
    cudaMalloc((void **)&d_C, C_size * num_batches * sizeof(float));
    // Memory copy from CPU to GPU
    cudaMemcpy(d_rpt, h_rpt.data(), (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cid, h_cid.data(), A_nnz * num_batches * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_val.data(), A_nnz * num_batches * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), B_size * num_batches * sizeof(float), cudaMemcpyHostToDevice);
    // --------------------------------------------------------------------------
    auto mcph2d_stop = chrono::high_resolution_clock::now();
    float mcph2d_time = chrono::duration<float, std::milli>(mcph2d_stop - mcph2d_start).count();
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            STEP TWO
            INITIALIZATION FOR CUSPARSE
        """
    */
    // create matrix descriptors
    auto init_start = chrono::high_resolution_clock::now();
    // --------------------------------------------------------------------------
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                      d_rpt, d_cid, d_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCsrSetStridedBatch(matA, num_batches, 0, A_nnz);
    // Create dense matrix B
    cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, d_B, CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseDnMatSetStridedBatch(matB, num_batches, B_size);
    // Create dense matrix C
    cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, d_C, CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseDnMatSetStridedBatch(matC, num_batches, C_size);
    // allocate an external buffer if needed
    cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_CSR_ALG2, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    // --------------------------------------------------------------------------
    auto init_stop = chrono::high_resolution_clock::now();
    float init_time = chrono::duration<float, std::milli>(init_stop - init_start).count();
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            STEP THREE
            GPU KERNEL COMPLEMENTATION
            BATCHED CSR-FORMAT SPMM
        """
    */
    // Warm-up
    for (int it = 0; it < 10; it++)
        cusparseSpMM(handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                 CUSPARSE_SPMM_CSR_ALG2, dBuffer);
    // --------------------------------------------------------------------------
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventRecord(kernel_start, 0);
    // --------------------------------------------------------------------------
    // Average on several implementations
    for (int it = 0; it < ITER_MAX; it++)
    cusparseSpMM(handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                 CUSPARSE_SPMM_CSR_ALG2, dBuffer);
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
    // Memory copy of SpMM results from GPU to CPU
    auto mcpd2h_start = chrono::high_resolution_clock::now();
    // --------------------------------------------------------------------------
    float *res = (float *)malloc(C_size * num_batches * sizeof(float));
    cudaMemcpy(res, d_C, C_size * num_batches * sizeof(float), cudaMemcpyDeviceToHost);
    // --------------------------------------------------------------------------
    auto mcpd2h_stop = chrono::high_resolution_clock::now();
    float mcpd2h_time = chrono::duration<float, std::milli>(mcpd2h_stop - mcpd2h_start).count();
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            STEP FIVE
            DESTRUCTION FOR CUSPARSE
        """
    */
    // destroy matrix descriptors
    auto des_start = chrono::high_resolution_clock::now();
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);
    auto des_end = chrono::high_resolution_clock::now();
    float des_time = chrono::duration<float, std::milli>(des_end - des_start).count();
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            STEP SIX
            FREE MEMORY
        """
    */
    auto free_start = chrono::high_resolution_clock::now();
    cudaFree(dBuffer);
    cudaFree(d_rpt);
    cudaFree(d_cid);
    cudaFree(d_val);
    cudaFree(d_B);
    cudaFree(d_C);
    // --------------------------------------------------------------------------
    auto free_stop = chrono::high_resolution_clock::now();
    float free_time = chrono::duration<float, std::milli>(free_stop - free_start).count();
    // --------------------------------------------------------------------------
    // Verify the SpMM results
    verify(res, h_C);
    // --------------------------------------------------------------------------
    // Free space
    h_C.clear();
    free(res);
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            REPORT OF RESULTS
        """
    */
    // Compute GFLOPS (Giga Floating-point Operations Per Second)
    float GFLOPS = FLOP * 1.0 / (kernel_time / ITER_MAX / 10e3) / 10e9;
    cout << "Batched CSR-format SpMM" << endl
         << "GFLOPS:\t"
         << fixed << setprecision(6)
         << GFLOPS << endl
         << "Time statistics (ms):" << endl
         << "cudaMalloc & cudaMemcpyHostToDevice\t" << mcph2d_time << endl
         << "cuSPARSE initialization\t" << init_time << endl
         << "CUDA kernel function\t" << kernel_time / ITER_MAX << endl
         << "cudaMemcpyDevicetoHost\t" << mcpd2h_time << endl
         << "cuSPARSE destruction\t" << des_time << endl
         << "cudaFree\t" << free_time << endl;
    return 0;
}