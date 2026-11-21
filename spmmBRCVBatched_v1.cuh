/*
    Initial version of batched BRCV-format sparse matrix - matrix multiplication
    Ideas proposed by Li LIU & Changbo CHEN
    Reference
        [1] Li LIU and Changbo CHEN. Band sparse matrix multiplication and efficient GPU implementation (in Chinese)[J]. Journal of Computer Applications, 2023, 43(12): 3856-3867.
        http://www.joca.cn/CN/10.11772/j.issn.1001-9081.2022111720
        [2] Li LIU. Efficient GPU implementation of band sparse matrix multiplication and applications (in Chinese)[D]. Beijing: University of Chinese Academy of Sciences, June 2023.
        http://dpaper.las.ac.cn/Dpaper/detail/detailNew?paperID=20217376&title=%E5%B8%A6%E7%8A%B6%E7%A8%80%E7%96%8F%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%E7%9A%84%E9%AB%98%E6%95%88GPU%E5%AE%9E%E7%8E%B0%E5%8F%8A%E5%BA%94%E7%94%A8&author=%E5%88%98%E4%B8%BD&highsearch=teacher_name%253A345345345%25E9%2599%2588%25E9%2595%25BF%25E6%25B3%25A2345345345&sortField=score%2520desc%252Cid&start=0&actionType=Browse&searchText=%E9%99%88%E9%95%BF%E6%B3%A2
    Re-coded and annotated by Yuxuan SONG
    Last modified time: Apr 19, 2024
*/
#define ILPMAX 4 // Instruction level parallelism
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
/*
    """
        STEP THREE
        GPU KERNEL COMPLEMENTATION
        BATCHED BRCV-FORMAT SPMM
    """
        (b, rc, rv, c, v):
            Concation of corresponding five arrays in BRCV-format A's
        (pb, prc, prv, pc, pv):
            Cumulative sum of corresponding lengths of arrays b's, rc's, rv's, c's, v's in BRCV-format A's
        B:
            Concation of corresponding 1-D dense-format arrays storing B's
        pB:
            Cumulative sum of corresponding numbers of elements in B's

        N:
            Concation of corresponding numbers of columns in B's
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
        C:
            Concation of corresponding 1-D dense-format arrays storing B's
        pC:
            Cumulative sum of corresponding numbers of elements in C's
*/
__global__ void BatBRCVSpMM(
    /*
        b:
            Concation of arrays b of BRCV-format A^{(i + 1)} (0 <= i <= L - 1)
        pb:
            Cumulative sum of the list of corresponding lengths of arrays b of A^{(i + 1)}
        b[pb[i] + j for j in range(p^{(i + 1)} + 1)]:
            Cumulative sum of the list of corresponding heights of p^{(i + 1)} sparse bands in A^{(i + 1)};
                if j == 0, then the value of the corresponding value of rv is 0.
    */
    const int *b, const int *pb,
    /*
        rc:
            Concation of arrays rc of BRCV-format A^{(i + 1)} (0 <= i <= L - 1)
        prc:
            Cumulative sum of the list of corresponding lengths of arrays rc of A^{(i + 1)}
        rc[prc[i] + j for j in range(p^{(i + 1)} + 1)]:
            Cumulative sum of the list of corresponding numbers of non-zero elements in the 1'st rows of p^{(i + 1)}
            sparse bands in A^{(i + 1)};
                if j == 0, then the value of the corresponding value of rv is 0.
    */
    const int *rc, const int *prc,
    /*
        rv:
            Concation of arrays rv of BRCV-format A^{(i + 1)} (0 <= i <= L - 1)
        prv:
            Cumulative sum of the list of corresponding lengths of arrays rv of A^{(i + 1)}
        rv[prv[i] + j for j in range(p^{(i + 1)})]:
            Cumulative sum of the list of corresponding numbers of non-zero elements of p^{(i + 1)} - 1 sparse bands in
            A^{(i + 1)};
                if j == 0, then the value of the corresponding value of rv is 0.
    */
    const int *rv, const int *prv,
    /*
        c:
            Concation of arrays c of BRCV-format A^{(i + 1)} (0 <= i <= L - 1)
        pc:
            Cumulative sum of the list of corresponding lengths of arrays c of A^{(i + 1)}
        c[pc[i] + j for j in range(Di)] (Di = rv[prv[i] + p - 1] + (b[pb[i] + p] - b[pb[i] + p - 1]) * (rc[prc[i] + p] - rc[prc[i] + p - 1])):
            Column indices of non-zero elements in the first row of p sparse bands in A^{(i + 1)}
    */
    const int *c, const int *pc,
    /*
        v:
            Concation of arrays v of BRCV-format A^{(i + 1)} (0 <= i <= L - 1)
        pv:
            Cumulative sum of the list of corresponding lengths of arrays v of A^{(i + 1)}
        v[pv[i] + j for j in range(nnz)]:
            Values of nnz non-zero elements in A^{(i + 1)}
    */
    const float *v, const int *pv,
    /*
        B:
            Concation of 1-D dense-format B^{(i + 1)} (0 <= i <= L - 1)
        pB:
            Cumulative sum of the list of corresponding numbers of elements of B^{(i + 1)}
    */
    const float *B, const int *pB,
    // Concation of corresponding numbers of columns in B^{(i + 1)} (0 <= i <= L - 1)
    const int *N,
    // Indices of sparse matrices assigned to y-axis GY groups of blocks on the grid
    const int *mb,
    // Indices of sparse bands in the correpsonding sparse matrices assigned to y-axis GY groups of blocks on the grid
    const int *bb,
    // Y-axis indices of sub sparse bands in the corresponding sparse bands assigned to y-axis GY groups of blocks on the grid
    const int *sb,
    // Numbers of non-zero elements in the 1'st row of the corresponding sparse band handed over to the workloads of y-axis GY groups of blocks on the grid
    const int *db,
    // Numbers of columns in dense matrices handed over to the workloads of y-axis GY groups of blocks on the grid
    const int *nb,
    /*
        C:
            Concation of 1-D dense-format C^{(i + 1)} (0 <= i <= L - 1)
        pC:
            Cumulative sum of the list of corresponding numbers of elements of C^{(i + 1)}
    */
    float *C, const int *pC)
{
    // Number of x-axis groups of blocks on the grid
    int BX = blockDim.x;
    // Number of y-axis groups of blocks on the grid
    int BY = blockDim.y;
    /*
        STEP 301:
            Fetch the values of load balancing parameters indexed by y-axis index of current block
    */
    // X-axis index of block alpha on the grid
    int bx = blockIdx.x;
    // Y-axis index of block alpha on the grid
    int by = blockIdx.y;
    // Index of the sparse matrix handed over to block alpha, 0 <= i <= L - 1
    int i = mb[by];

    // Index of the sparse band in the (i + 1)'th sparse matrix handed over to block alpha, 0 <= j <= p^{(i + 1)} - 1
    int j = bb[by];

    // Y-axis index of sub sparse band in the (j + 1)'th sparse band handed over to block alpha
    int k = sb[by];

    // Number of rows in the (i + 1)'th dense matrix accessed in the shared memory of block alpha
    int Da = db[by];

    // Number of columns in the (i + 1)'th dense matrix accessed in the shared memory of block alpha
    int Na = nb[by];

    // Number of columns in the (i + 1)'th dense matrix
    int Ni = N[i];
    /*
        Offset of the column index of the 1'st column within Na columns in B^{(i + 1)} handed over to
        the workload of block alpha in an array of all N^{(i + 1)} column indices
    */
    int BcidSt = bx * Na;
    // Starting places in arrays b, rc, c, v representing BRCV-format A^{(i + 1)}
    int sp_b = pb[i], sp_rc = prc[i], sp_rv = prv[i], sp_c = pc[i], sp_v = pv[i];

    // Starting places in arrays B, C representing B^{(i + 1)}, C^{(i + 1)}
    int sp_B = pB[i], sp_C = pC[i];
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        STEP 302
    */
    // Height of the (j + 1)'th sparse band in A^{(i + 1)}
    int Hj = b[sp_b + j + 1] - b[sp_b + j];

    // Number of non-zero elements in the 1'st row of the (j + 1)'th sparse band in A^{(i + 1)}
    int Dj = rc[sp_rc + j + 1] - rc[sp_rc + j];
    // --------------------------------------------------------------------------
    // --------------------------------------------------------------------------
    /*
        STEP 303
    */
    // X-axis index of thread pi on block alpha
    int tx = threadIdx.x;
    // Y-axis index of thread pi on block alpha
    int ty = threadIdx.y;
    /*
        Offset of the row index in A^{(i + 1)} handed over to thread pi
        in an array of Hj row indices in the (j + 1)'th sparse band
    */
    int AridHj = k * BY + ty;
    /*
        Scale of instruction-level parallelism,
        which is the number of elements in C^{(i + 1)} simultaneously calcultaed by block alpha
    */
    int ILP = (Na + BX - 1) / BX;
    /*
        Zero-initialized register array to accumulate intermediate SpMM results
        The maximum size of register array has to be defined as a constant (#define ILPMAX 4)
    */
    float res[ILPMAX] = {0.0};
    // --------------------------------------------------------------------------
    // Declare the shared memory in block alpha
    extern __shared__ int sh_mem[];

    // Column indices of Da columns of non-zero elements in A^{(i + 1)}
    int *Acol = sh_mem;
    // Values of Da columns, BY rows of non-zero elements in A^{(i + 1)}
    float *Aval = (float *)&sh_mem[Da];

    // Values of corresponding Da rows, Na columns of elements in B^{(i + 1)}
    float *Bval = &Aval[BY * Da];
    // --------------------------------------------------------------------------
    /*
        Offset of the column index of one non-zero element in the first row of the (j + 1)'th sparse band
        set by thread pi in an array of column indices of all Dj non-zero elements in the first row
        of the (j + 1)'th sparse band
    */
    int AcidDj;
    /*
        Offset of the row index of one element in B^{(i + 1)} set by thread pi
        in the array Acol in the shared memory
    */
    int BridAc;
    /*
        One same row index of several elements in B^{(i + 1)} set by thread pi
    */
    int Brid;
    /*
        Offset of the column index of one non-zero element set by thread pi
        in an array of all Na column indices of B^{(i + 1)} handed over to block alpha
    */
    int BcidNa;
    /*
        Offset of the column index of one non-zero element set by thread pi
        in an array of all N^{(i + 1)} column indices of B^{(i + 1)}
    */
    int BcidNi;
    int vmax;
    if (AridHj < Hj && BcidSt < Ni)
    /*
        Boundary check to prevent block alpha from processing different sparse bands &
        to ensure the positions of elements in B^{(i + 1)} are legal
    */
    {
        /*
            STEP 304:
                \ceil{D_j/D_{\alpha}} times of resetting shared memory and
                conducting matrix multiplication operations via the shared memory
        */
        for (int r = 0; r < Dj; r += Da)
        {
            // --------------------------------------------------------------------------
            // --------------------------------------------------------------------------
            /*
                STEP 305
                    ((r + Da) / Da)'th resetting shared memory of Acol & Aval
            */
            AcidDj = r + tx;
            if (tx < Da && AcidDj < Dj)
            /*
                Boundary check to avoid overwriting elements in wrong region in shared memory &
                to ensure the positions of column indices of non-zero elements
                    in the first row of the (j + 1)'th sparse band are legal
            */
            {
                Acol[tx] = c[sp_c + rc[sp_rc + j] + AcidDj];
                Aval[ty * Da + tx] = v[sp_v + rv[sp_rv + j] + AridHj * Da + AcidDj];
            }
            __syncthreads();
            // --------------------------------------------------------------------------
            // --------------------------------------------------------------------------
            /*
                STEP 306
                    ((r + Da) / Da)'th resetting shared memory of Bval
            */
            for (int s = 0; s < Da; s += min(BY, Hj - AridHj))
            /*
                All threads in block alpha set the Da rows of elements in B^{(i + 1)} with a step of BY
                Adding min on the step satisfies the scenario of BY > Da > Hj - AridHj
            */
            {
                BridAc = ty + s;
                if (BridAc < Da && r + BridAc < Dj)
                /*
                    Boundary check to ensure fetching elements in correct region in shared memory &
                    to ensure the row indices of B^{(i + 1)} corresponding to their column couterparts
                        in the first row of the (j + 1)'th sparse band are legal
                */
                {
                    Brid = Acol[BridAc]; // Row index of B^{(i + 1)}
                    for (int t = 0; t < Na; t += BX)
                    {
                        BcidNa = tx + t;
                        BcidNi = BcidSt + BcidNa;
                        if (BcidNa < Na && BcidNi < Ni)
                            /*
                                Boundary check to ensure fetching elements in correct region in shared memory &
                                to ensure the column indices are within the scope of B^{(j + 1)}
                            */
                            Bval[BridAc * Na + BcidNa] = B[sp_B + Brid * Ni + BcidNi];
                    }
                }
            }
            __syncthreads();
            // --------------------------------------------------------------------------
            // --------------------------------------------------------------------------
            /*
                STEP 307
                    Conduct matrix multiplication operations via
                    exploiting ((r + Da) / Da)'th reset shared memory
            */
            vmax = min(Da, Dj - r); // Necessary for the last iteration if Dj is not divisible by Da
            for (int p = 0; p < ILP; p++)
            {
                for (int v = 0; v < vmax; v++)
                {
                    BcidNa = tx + p * BX;
                    if (BcidNa < Na)
                        res[p] += Aval[ty * Da + v] * Bval[v * Na + BcidNa];
                }
            }
            __syncthreads();
            // --------------------------------------------------------------------------
            // --------------------------------------------------------------------------
            /*
                STEP 308
                    Set the value of the element in C calculated by the block by values in array res
            */
            for (int w = 0; w < Na; w += BX)
            {
                BcidNa = tx + w;
                BcidNi = BcidSt + BcidNa;
                if (BcidNa < Na && BcidNi < Ni)
                    /*
                        Boundary check to ensure fetching elements in correct region in shared memory &
                        to ensure the column indices are within the scope of B^{(j + 1)}
                    */
                    C[sp_C + (b[sp_b + j] + AridHj) * Ni + BcidNi] = res[w / BX];
            }
        }
    }
}