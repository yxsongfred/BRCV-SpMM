/*
    Initial version of batched BRCV-format sparse matrix - matrix multiplication
    Ideas proposed by Li LIU & Changbo CHEN
    Reference
        [1] Li LIU and Changbo CHEN. Band sparse matrix multiplication and efficient GPU implementation (in Chinese)[J]. Journal of Computer Applications, 2023, 43(12): 3856-3867.
        http://www.joca.cn/CN/10.11772/j.issn.1001-9081.2022111720
        [2] Li LIU. Efficient GPU implementation of band sparse matrix multiplication and applications (in Chinese)[D]. Beijing: University of Chinese Academy of Sciences, June 2023.
        http://dpaper.las.ac.cn/Dpaper/detail/detailNew?paperID=20217376&title=%E5%B8%A6%E7%8A%B6%E7%A8%80%E7%96%8F%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%E7%9A%84%E9%AB%98%E6%95%88GPU%E5%AE%9E%E7%8E%B0%E5%8F%8A%E5%BA%94%E7%94%A8&author=%E5%88%98%E4%B8%BD&highsearch=teacher_name%253A345345345%25E9%2599%2588%25E9%2595%25BF%25E6%25B3%25A2345345345&sortField=score%2520desc%252Cid&start=0&actionType=Browse&searchText=%E9%99%88%E9%95%BF%E6%B3%A2
    Re-coded and annotated by Yuxuan SONG
    Last modified time: Apr 27, 2024
    Precious experiences when debugging
        1. When the "printf" function is used to print some intermediate results, one sentence must be attached to
        the position right after the call of kernel function:
            cudaDeviceSynchronize();
        2. Method of detecting non-logical errors in CUDA kernel function, which is to attach the following sentences
        right after the call of kernel function:
            cudaError_t err = cudaGetLastError();
            cout << cudaGetErrorString(err) << endl;
        3. If the buffer zone requested by the "printf" function in CUDA kernel function exceeds the maximum buffer
        size (normally 1 metabytes), an alternation of the buffer size can be done by:
            cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 10);
*/
/*
    Prefix illustration:
        "tX" is a temporary variable utilized to obtain variable X
        "h_X" is X on host (CPU)
        "d_X" is X on device (GPU)
*/
#include <iostream>
#include <vector>
#include "utils.h"
#include <chrono>
#include <algorithm>
#include "spmmBRCVBatched_v1.cuh"
#include <cuda_runtime.h>

#define KSAI 512 // KSAI is a preset size of 2-D block on GPU
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

    string folderdir, filepfx;
    // Obtain the directory of dataset folder according to the input parameter
    datasetMap(argc, argv, folderdir);
    vector<int> idx;
    readcsv(folderdir + "indices.csv", idx);
    /*
        L denotes L groups of SpMM's conducted simultaneously in GPU
        Each group is consisted of a sparse matrix A^{(i + 1)} and two dense matrices B^{(i + 1)}, C^{(i + 1)},
            where 0 <= i <= L - 1
    */
    int L = idx.size() / 4;
    /*
        rpt[M + 1]
            Row pointer
            rpt[i] indicates the sum of nnz's in the first i'th rows
    */
    vector<int> rpt;
    /*
        cid[nnz]: column IDs of non-zero elements
    */
    vector<int> cid;
    /*
        val[nnz]: values of non-zero elements
    */
    vector<float> val;
    // Temporary variable utilized to obtain h_b, h_rc, h_rv, h_c
    vector<int> tb, trc, trv, tc;
    // Temporary variable utilized to obtain h_v, h_B, h_C
    vector<float> tv, tB, tC;
    // Concation of corresponding arrays b, rc, rv, c in BRCV-format A's
    vector<int> h_b, h_rc, h_rv, h_c;
    // Concation of corresponding array v in BRCV-format A's
    vector<float> h_v;
    // Concation of corresponding 1-D dense-format arrays storing B's and C's
    vector<float> h_B, h_C;
    // Cumulative sum of corresponding lengths of arrays b's, rc's, rv's, c's, v's in BRCV-format A's
    vector<int> h_pb, h_prc, h_prv, h_pc, h_pv;
    // Cumulative sum of corresponding numbers of elements in B's and C's
    vector<int> h_pB, h_pC;
    // Concation of corresponding numbers of columns in B's
    vector<int> h_N;
    // Number of Floating-point Operations
    long long FLOP = 0;
    // An initialization of the 0'th place is profitable
    h_pb.push_back(0);
    h_prc.push_back(0);
    h_prv.push_back(0);
    h_pc.push_back(0);
    h_pv.push_back(0);
    h_pB.push_back(0);
    h_pC.push_back(0);
    // --------------------------------------------------------------------------
    // Traverse L groups
    for (int i = 0; i < idx.size(); i += 4)
    {
        /*
            Obtain dimension parameters:
                idx[i] = M^{(i / 4 + 1)}
                idx[i + 1] = N^{(i / 4 + 1)}
                idx[i + 2] = K^{(i / 4 + 1)}
                idx[i + 3] = i / 4 + 1
            A^{(i / 4 + 1)}\in\mathbb{R}^{M^{(i / 4 + 1)}\times K^{(i / 4 + 1)}}
            B^{(i / 4 + 1)}\in\mathbb{R}^{K^{(i / 4 + 1)}\times N^{(i / 4 + 1)}}
            C^{(i / 4 + 1)}\in\mathbb{R}^{M^{(i / 4 + 1)}\times N^{(i / 4 + 1)}}
        */
        // File name prefix: "M[i]_N[i]_K[i]_i_"
        filepfx = folderdir + to_string(idx[i]) + "_" + to_string(idx[i + 1]) + "_" + to_string(idx[i + 2]) + "_" + to_string(idx[i + 3]) + "_";
        readcsv(filepfx + "rpt.csv", rpt);
        readcsv(filepfx + "cid.csv", cid);
        readcsv(filepfx + "val.csv", val);
        readcsv(filepfx + "b.csv", tb);
        readcsv(filepfx + "B.csv", tB);
        readcsv(filepfx + "C.csv", tC);
        // Conduct the naiive conversion from CSR format to BRCV format for A's
        csr2brcvNaiive(rpt, cid, val, tb, trc, trv, tc, tv);
        rpt.clear();
        cid.clear();
        val.clear();
        // Set the values in pointer arrays
        if (i / 4 < L - 1)
        {
            h_pb.push_back(h_pb[i / 4] + tb.size());
            h_prc.push_back(h_prc[i / 4] + trc.size());
            h_prv.push_back(h_prv[i / 4] + trv.size());
            h_pc.push_back(h_pc[i / 4] + tc.size());
            h_pv.push_back(h_pv[i / 4] + tv.size());
            h_pB.push_back(h_pB[i / 4] + tB.size());
            h_pC.push_back(h_pC[i / 4] + tC.size());
        }
        FLOP += tv.size() * idx[i + 1] * 2; // idx[i + 1] = N^{(i)}
        // Concation operations in arrays b, rc, rv, c, v, B, C
        vecAppend(tb, h_b);
        tb.clear();
        vecAppend(trc, h_rc);
        trc.clear();
        vecAppend(trv, h_rv);
        trv.clear();
        vecAppend(tc, h_c);
        tc.clear();
        vecAppend(tv, h_v);
        tv.clear();
        vecAppend(tB, h_B);
        tB.clear();
        vecAppend(tC, h_C);
        tC.clear();
        h_N.push_back(idx[i + 1]);
    }
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            STEP ONE
            INITIALIZATION FOR LOAD-BALANCE PRINCIPLES
            WORK DONE ON CPU
        """
    */
    /*
        BRCV format:
            Statement on non-zero elements in the current context:
                they don't strictly refer to elements with absolute non-zero values
                while they are interpreted as elements to be treated as non-zero elements instead.
            Abbreviation illustration:
                D stands for the number of non-zero elements in BRCV format
                    which is derived from Portuguese "elemntos differentes de zero"
                        D for "differentes" ("other than" in Engish)
                p stands for the number of sparse bands
                H stands for the height of a sparse band, which is the sum of rows in a sparse band
            b[p + 1]:
                sum of H's of sparse bands
                b[i] indicates the sum of H's in the first i'th sparse bands
            rc[p + 1]:
                sum of D's in the first rows of sparse bands,
                rc[i] indicates the sum of D's in the first rows of the first i'th sparse bands
            rv[p]:
                sum of D's in sparse bands
                rv[i] indicates the sum of D's in the first i'th sparse bands
            c[rv[p - 1] + (b[p] - b[p - 1]) * (rc[p] - rc[p - 1])]:
                column IDs of non-zero elements in the first row
            v[nnz]:
                values of non-zero elements
    */
    auto init_start = chrono::high_resolution_clock::now();
    // --------------------------------------------------------------------------
    /*
        STEP 101: traverse all the sparse bands in all sparse matrices and dense matrices
            with a purpose of obtaining H's & D's in the first row of all sparse bands
    */
    // H stands for the height of a sparse band, which is the sum of rows in a sparse band
    vector<int> H;
    /*
        D stands for the number of non-zero elements in BRCV format
            which is derived from Portuguese "elemntos differentes de zero"
            D for "differentes" ("other than" in Engish)
    */
    vector<int> D;

    for (int i = 0; i < L; i++)
    {
        // Compute the H and D of the (j + 1)'th sparse band in A^{(i + 1)}
        for (int j = h_pb[i]; (i < L - 1 && j < h_pb[i + 1] - 1) || (i == L - 1 && j < h_b.size() - 1); j++)
        /*
            (h_pb[i])'th, ..., (h_pb[i + 1] - 1)'th positions in h_b store array b of A^{(i + 1)} (1 <= i < L)
            (h_pb[L - 1])'th, ..., (h_b.size() - 1)'th positions in h_b store array b of A^{(L)}
            The last position of array b should be noted,
                where j cannot outnumber h_pb[i + 1] - 2 or h_pb.size() - 2
                for that the length of array b is equal to #(sparse bands) plus 1
        */
        {
            H.push_back(h_b[j + 1] - h_b[j]);
            D.push_back(h_rc[j + 1] - h_rc[j]);
        }
    }
    /*
        STEP 102: find minimum in H's, minimum and maximum in D's, maximum in N's
    */
    int Hmin = *min_element(H.begin(), H.end());
    int Dmin = *min_element(D.begin(), D.end());
    int Dmax = *max_element(D.begin(), D.end());
    int Nmax = *max_element(h_N.begin(), h_N.end());
    /*
        STEP 103: determine the values of BX, BY & GX
    */

    /*
        BX, BY denote there are BX x-axis(column) threads in each of BY y-axis (row) threads
        in one block, and therefore there are BX * BY threads on a 2-D block
    */
    int BY = min(int(pow(2, ceil(log2(Hmin)))), 32), BX = KSAI / BY;
    /*
        GX, GY denote there are GX x-axis(column) blocks in each of GY y-axis (row) blocks
        in one grid, and therefore there are GX * GY blocks on a 2-D grid
    */
    int GX = (Nmax + BX - 1) / BX, GY = 0;
    /*
        STEP 104: obtain the value of GY and arrays mb, bb, sb, db, nb, which are parameters
            assigned to each block in GY blocks in y-axis of the grid
            considering:
                1) load-balance formulas
                2) reduction of idle threads as many as possible
                3) avoidance of handling elements from different sparse bands in one block
    */

    /*
        (mb, bb, sb, db, nb):
            Load balancing parameters assigned to y-axis GY groups of blocks on the grid
            mb:
                Indices of sparse matrices
            bb:
                Indices of sparse bands in the corresponding sparse matrices
            sb:
                Y-axis indices of sub sparse bands in the corresponding sparse bands
            db:
                Numbers of non-zero elements in the 1'st row in the corresponding sparse bands handed over to the workloads of blocks
            nb:
                Numbers of columns in dense matrices handed over to the workloads of blocks
    */
    vector<int> h_mb, h_bb, h_sb, h_db, h_nb;
    // Number of y-axis blocks required in processing one current sparse band
    int Q;
    int Da, Na, jm;
    int f = Dmin > BY ? minFactorNgeK(Dmin, BY) : Dmin;
    int it = 0;
    for (int i = 0; i < L; i++)
    {
        jm = h_pb[i];
        for (int j = jm; (i < L - 1 && j < h_pb[i + 1] - 1) || (i == L - 1 && j < h_b.size() - 1); j++)
        // Traverse each sparse band
        {
            Q = (H[it] + BY - 1) / BY;
            GY += Q;
            Da = (f * D[it] + Dmin - 1) / Dmin;
            Na = min((BX * Dmax + D[it] - 1) / D[it], h_N[i]);
            for (int k = 0; k < Q; k++)
            {
                h_mb.push_back(i);
                h_bb.push_back(j - jm);
                h_sb.push_back(k);
                h_db.push_back(Da);
                h_nb.push_back(Na);
            }
            it++;
        }
    }
    // Declare the sizes of grid and block
    dim3 grid(GX, GY);
    dim3 block(BX, BY);
    // --------------------------------------------------------------------------
    auto init_stop = chrono::high_resolution_clock::now();
    double init_time = chrono::duration<double, std::milli>(init_stop - init_start).count();
    // --------------------------------------------------------------------------
    H.clear();
    D.clear();
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
    int *d_b, *d_rc, *d_rv, *d_c, *d_pb, *d_prc, *d_prv, *d_pc, *d_pv, *d_pB, *d_N, *d_pC;
    int *d_mb, *d_bb, *d_sb, *d_db, *d_nb;
    float *d_v, *d_B, *d_C;
    cudaMalloc((void **)&d_b, h_b.size() * sizeof(int));
    cudaMalloc((void **)&d_rc, h_rc.size() * sizeof(int));
    cudaMalloc((void **)&d_rv, h_rv.size() * sizeof(int));
    cudaMalloc((void **)&d_c, h_c.size() * sizeof(int));
    cudaMalloc((void **)&d_v, h_v.size() * sizeof(float));
    cudaMalloc((void **)&d_B, h_B.size() * sizeof(float));
    cudaMalloc((void **)&d_pb, h_pb.size() * sizeof(int));
    cudaMalloc((void **)&d_prc, h_prc.size() * sizeof(int));
    cudaMalloc((void **)&d_prv, h_prv.size() * sizeof(int));
    cudaMalloc((void **)&d_pc, h_pc.size() * sizeof(int));
    cudaMalloc((void **)&d_pv, h_pv.size() * sizeof(int));
    cudaMalloc((void **)&d_pB, h_pB.size() * sizeof(int));
    cudaMalloc((void **)&d_N, h_N.size() * sizeof(int));
    cudaMalloc((void **)&d_pC, h_pC.size() * sizeof(int));
    cudaMalloc((void **)&d_mb, h_mb.size() * sizeof(int));
    cudaMalloc((void **)&d_bb, h_bb.size() * sizeof(int));
    cudaMalloc((void **)&d_sb, h_sb.size() * sizeof(int));
    cudaMalloc((void **)&d_db, h_db.size() * sizeof(int));
    cudaMalloc((void **)&d_nb, h_nb.size() * sizeof(int));
    cudaMalloc((void **)&d_C, h_C.size() * sizeof(float));
    // Memory copy from CPU to GPU
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rc, h_rc.data(), h_rc.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rv, h_rv.data(), h_rv.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c.data(), h_c.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pb, h_pb.data(), h_pb.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prc, h_prc.data(), h_prc.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prv, h_prv.data(), h_prv.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pc, h_pc.data(), h_pc.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pv, h_pv.data(), h_pv.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pB, h_pB.data(), h_pB.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N.data(), h_N.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pC, h_pC.data(), h_pC.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mb, h_mb.data(), h_mb.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bb, h_bb.data(), h_bb.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sb, h_sb.data(), h_sb.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_db, h_db.data(), h_db.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb, h_nb.data(), h_nb.size() * sizeof(int), cudaMemcpyHostToDevice);
    // --------------------------------------------------------------------------
    auto mcph2d_stop = chrono::high_resolution_clock::now();
    float mcph2d_time = chrono::duration<float, std::milli>(mcph2d_stop - mcph2d_start).count();
    // --------------------------------------------------------------------------
    // Clear vectors no longer useful
    h_b.clear();
    h_rc.clear();
    h_rv.clear();
    h_c.clear();
    h_v.clear();
    h_B.clear();
    h_pb.clear();
    h_prc.clear();
    h_prv.clear();
    h_pc.clear();
    h_pv.clear();
    h_pB.clear();
    h_N.clear();
    h_pC.clear();
    h_mb.clear();
    h_bb.clear();
    h_sb.clear();
    h_db.clear();
    h_nb.clear();
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        """
            STEP THREE
            GPU KERNEL COMPLEMENTATION
            BATCHED BRCV-FORMAT SPMM
        """
    */
    // Warm-up
    for (int it = 0; it < 10; it++)
        BatBRCVSpMM<<<grid, block, 64 * sizeof(int) + BY * 64 * sizeof(float) + 64 * Nmax * sizeof(float)>>>(d_b, d_pb, d_rc, d_prc, d_rv, d_prv, d_c, d_pc, d_v, d_pv, d_B, d_pB, d_N, d_mb, d_bb, d_sb, d_db, d_nb, d_C, d_pC);
    cudaDeviceSynchronize();
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventRecord(kernel_start, 0);
    // --------------------------------------------------------------------------
    // Average on several implementations
    for (int it = 0; it < ITER_MAX; it++)
        /*
            Variable size of shared memory for each block
            Preset maximum size of shared memory:
                Acol: Da: Da_{max} = 64
                Aval: BY * Da : BY * Da_{max} = BY * 64
                Bval: Da * N: Da_{max} * Nmax = 64 * Nmax
        */
        // Call the kernel function BatBRCVSpMM1D
        BatBRCVSpMM<<<grid, block, 64 * sizeof(int) + BY * 64 * sizeof(float) + 64 * Nmax * sizeof(float)>>>(d_b, d_pb, d_rc, d_prc, d_rv, d_prv, d_c, d_pc, d_v, d_pv, d_B, d_pB, d_N, d_mb, d_bb, d_sb, d_db, d_nb, d_C, d_pC);
    cudaDeviceSynchronize();
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
    float *res = (float *)malloc(h_C.size() * sizeof(float));
    cudaMemcpy(res, d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);
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
    // --------------------------------------------------------------------------
    cudaFree(d_b);
    cudaFree(d_rc);
    cudaFree(d_rv);
    cudaFree(d_c);
    cudaFree(d_v);
    cudaFree(d_B);
    cudaFree(d_pb);
    cudaFree(d_prc);
    cudaFree(d_prv);
    cudaFree(d_pc);
    cudaFree(d_pv);
    cudaFree(d_pB);
    cudaFree(d_N);
    cudaFree(d_pC);
    cudaFree(d_mb);
    cudaFree(d_bb);
    cudaFree(d_sb);
    cudaFree(d_C);
    // --------------------------------------------------------------------------
    auto free_stop = chrono::high_resolution_clock::now();
    float free_time = chrono::duration<float, std::milli>(free_stop - free_start).count();
    // --------------------------------------------------------------------------
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
    cout << "Batched BRCV-format SpMM" << endl
         << "GFLOPS:\t"
         << fixed << setprecision(6)
         << GFLOPS << endl
         << "Time statistics (ms):" << endl
         << "Initialization for load-balance principles\t" << init_time << endl
         << "cudaMalloc & cudaMemcpyHostToDevice\t" << mcph2d_time << endl
         << "CUDA kernel function\t" << kernel_time / ITER_MAX << endl
         << "cudaMemcpyDevicetoHost\t" << mcpd2h_time << endl
         << "cudaFree\t" << free_time << endl;
    return 0;
}