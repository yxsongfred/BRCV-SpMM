/*
    A serial implementation of batched CSR-format sparse matrix-matrix multiplication accelerated by cuSPARSE
    Ideas inspired by NVIDIA(R) CUDALibrarySamples
    Reference
        [1] cuSPARSE Generic APIs - cusparseSpMM CSR
        https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spmm_csr/spmm_csr_example.c
    Re-coded and annotated by Yuxuan SONG
    Last modified time: May 17, 2024
*/
#include <iostream>
#include <vector>
#include "utils.h"
#include <chrono>
#include <cuda_runtime.h>
#include <cusparse.h>

using namespace std;

int main(int argc, char *argv[])
{
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
    // CSR-format data loader
    vector<int> h_rpt, h_cid;
    vector<float> h_val;
    vector<float> tB, h_B, tC, h_C;
    float mcph2d_time = 0, init_time = 0, kernel_time = 0, mcpd2h_time = 0, des_time = 0, free_time = 0;
    // Number of Floating-point Operations
    long long FLOP = 0;
    const float alpha = 1.0f;
    const float beta = 0.0f;
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
        filepfx = folderdir + to_string(idx[i]) + "_" + to_string(idx[i + 1]) + "_" + to_string(idx[i + 2]) + "_" + to_string(idx[i + 3]) + "_";
        readcsv(filepfx +"cid.csv", h_cid);
        readcsv(filepfx + "_" + "rpt.csv", h_rpt);
        readcsv(filepfx + "_" + "val.csv", h_val);
        readcsv(filepfx + "_" + "B.csv", tB);
        readcsv(filepfx + "_" + "C.csv", tC);
        /*
            VITAL!!!! Dense matrices must be transposed!!!
            Convert row-major storation to column-major storation
        */
        mtxTranspose(tB, h_B, idx[i + 2]);
        tB.clear();
        mtxTranspose(tC, h_C, idx[i]);
        tC.clear();
        // --------------------------------------------------------------------------
        // Host problem definition
        int A_num_rows = idx[i], A_num_cols = idx[i + 2], A_nnz = h_val.size();
        int B_num_rows = A_num_cols, B_num_cols = idx[i + 1];
        int ldb = B_num_rows, ldc = A_num_rows;
        int B_size = ldb * B_num_cols, C_size = ldc * B_num_cols;
        // --------------------------------------------------------------------------
        FLOP += h_val.size() * B_num_cols * 2;
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
        cudaMalloc((void **)&d_cid, A_nnz * sizeof(int));
        cudaMalloc((void **)&d_val, A_nnz * sizeof(float));
        cudaMalloc((void **)&d_B, B_size * sizeof(float));
        cudaMalloc((void **)&d_C, C_size * sizeof(float));
        // Memory copy from CPU to GPU
        cudaMemcpy(d_rpt, h_rpt.data(), (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cid, h_cid.data(), A_nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, h_val.data(), A_nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), B_size * sizeof(float), cudaMemcpyHostToDevice);
        // --------------------------------------------------------------------------
        auto mcph2d_stop = chrono::high_resolution_clock::now();
        mcph2d_time += chrono::duration<float, milli>(mcph2d_stop - mcph2d_start).count();
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
        // Create dense matrix B
        cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, d_B, CUDA_R_32F, CUSPARSE_ORDER_COL);
        // Create dense matrix C
        cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, d_C, CUDA_R_32F, CUSPARSE_ORDER_COL);
        // allocate an external buffer if needed
        cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
        cudaMalloc(&dBuffer, bufferSize);
        // --------------------------------------------------------------------------
        auto init_stop = chrono::high_resolution_clock::now();
        init_time += chrono::duration<float, milli>(init_stop - init_start).count();
        // --------------------------------------------------------------------------
        // --------------------------------------------------------------------------
        /*
            """
                STEP THREE
                GPU KERNEL COMPLEMENTATION
                SINGLE CSR-FORMAT SPMM
            """
        */
        cudaEvent_t kernel_start, kernel_stop;
        cudaEventCreate(&kernel_start);
        cudaEventCreate(&kernel_stop);
        cudaEventRecord(kernel_start, 0);
        // --------------------------------------------------------------------------
        cusparseSpMM(handle,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                     CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
        // --------------------------------------------------------------------------
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
        // Memory copy of SpMM results from GPU to CPU
        auto mcpd2h_start = chrono::high_resolution_clock::now();
        // --------------------------------------------------------------------------
        float *res = (float *)malloc(C_size * sizeof(float));
        cudaMemcpy(res, d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost);
        // --------------------------------------------------------------------------
        auto mcpd2h_stop = chrono::high_resolution_clock::now();
        mcpd2h_time += chrono::duration<float, milli>(mcpd2h_stop - mcpd2h_start).count();
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
        des_time += chrono::duration<float, milli>(des_end - des_start).count();
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
        free_time += chrono::duration<float, milli>(free_stop - free_start).count();
        // --------------------------------------------------------------------------
        // Verify the SpMM results
        verify(res, h_C);
        // --------------------------------------------------------------------------
        // Free space
        h_C.clear();
        free(res);
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
    cout << "Serial CSR-format SpMM" << endl
         << "GFLOPS:\t"
         << fixed << setprecision(6)
         << GFLOPS << endl
         << "Time statistics (ms):" << endl
         << "cudaMalloc & cudaMemcpyHostToDevice\t" << mcph2d_time << endl
         << "cuSPARSE initialization\t" << init_time << endl
         << "CUDA kernel function\t" << kernel_time << endl
         << "cudaMemcpyDevicetoHost\t" << mcpd2h_time << endl
         << "cuSPARSE destruction\t" << des_time << endl
         << "cudaFree\t" << free_time << endl;
    return 0;
}