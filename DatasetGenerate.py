"""
    Generate the experimental dataset
    Coded and annotated by Yuxuan SONG
    Last modified time: Apr 19, 2024
"""
import os
import numpy as np
import random
from scipy.sparse import csr_matrix
from itertools import chain


def folderGen(_folderdir: str) -> str:
    """
        Return the directory of given folder, created first if not existed before
    """
    if os.path.exists(_folderdir) is False:
        os.makedirs(_folderdir)
    return _folderdir


"""
    CSR format:
        Premise: a SpM with a dimension of M * K
            Abbreviation illustration:
                nnz stands for the number of non-zero elements
                unless otherwise specified, nnz is counted in a whole SpM
            rpt[M + 1]:
                row pointer
                rpt[i] indicates the sum of nnz's in the first i'th rows
            cid[nnz]:
                column IDs of non-zero elements
            val[nnz]:
                values of non-zero elements
    
    BRCV format:
        Statement on non-zero elements in the current context:
            they don't strictly refer to elements with absolute non-zero values
            while they are interpreted as elements to be treated as non-zero elements instead.
        Abbreviation illustration:
            D stands for the number of non-zero elements in BRCV format
                which is derived from Portuguese 'elemntos differentes de zero'
                    D for 'differentes' ('other than' in Engish)
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
        c[rv[p - 1] + rc[p] * (b[p] - b[p - 1])]:
            column IDs of non-zero elements in the first row
        v[nnz]:
            values of non-zero elements
"""


def datasetEqMNKrptGenSave(L: int, M: int, N: int, K: int, Harr: list[int], Darr: list[int], no: int) -> None:
    """
        Generate dataset where each instances have the same (M, N, K) and
        every sparse matrix A has the same rpt array in CSR format

    Args:
        L: #batches
        M: #rows of SpM A & DeM C
        N: #columns of DeM's B & C
        K: #columns of SpM A & #rows of DeM B
        Harr: an array of #(rows)'s of all sparse bands
        Darr: an array of #(non-zero elements in the first row)'s of all sparse bands
        no: number to distinguish different datasets with the same M, N, K, rpt
    """
    folderdir = folderGen('./data/eqMNKrpt/M' + str(M) +
                          '_N' + str(N) + '_K' + str(K) + "/" + str(no))
    idx = []
    pfx = '/' + str(M) + '_' + str(N) + '_' + str(K) + '_'
    b = [0]
    rpt = [[0]]
    # Number of non-zero elements in the former several sparse bands
    fsp = 0
    # Generate array b in BRCV format & rpt in CSR format
    for i in range(len(Harr)):
        trpt = []
        b.append(b[i] + Harr[i])
        trpt = [fsp + Darr[i] * k for k in range(1, Harr[i] + 1)]
        rpt.append(trpt)
        fsp = trpt[-1]
    nnz = fsp   # number of non-zero elements in SpM A
    # Remove inner brackets [[,],[,,],[,,,]] --> [,,,,,,,,]
    rpt = list(chain.from_iterable(rpt))
    rpt = np.array(rpt)
    # Generate and store L groups of SpM A's, DeM B's & DeM C's
    for i in range(L):
        idx.append([M, N, K, i])
        filepfx = pfx + str(i) + '_'
        # Save arrays b & rpt of SpM A
        np.savetxt(folderdir + filepfx + 'b.csv', b, '%d', ',')
        np.savetxt(folderdir + filepfx + 'rpt.csv', rpt, '%d', ',')
        # Generate and save array val of SpM A
        val = np.random.randn(1, nnz)
        val = np.array(val.tolist()[0])
        np.savetxt(folderdir + filepfx + 'val.csv', val, '%f', ',')
        # Generate and save DeM B
        B = np.matrix(np.random.randn(K, N)).astype('float32')
        np.savetxt(folderdir + filepfx + 'B.csv', B, '%f', ',')
        # Generate and save array cid of SpM A
        cid = []
        for j in range(len(Harr)):
            tcid = random.sample([k for k in range(K)], Darr[j])
            tcid.sort()
            tcid = np.array(tcid * Harr[j])
            cid.append(tcid)
        cid = list(chain.from_iterable(cid))
        cid = np.array(cid)
        np.savetxt(folderdir + filepfx + 'cid.csv', cid, '%d', ',')
        # Generate CSR-format SpM A to obtain DeM C
        A = csr_matrix((val, cid, rpt), shape=(M, K)).astype('float32')
        C = A * B
        np.savetxt(folderdir + filepfx + 'C.csv', C, '%f', ',')
    np.savetxt(folderdir + '/indices.csv', idx, '%d', ',')


"""
    M = 32, N = 8/16, K = 32
    Harr = {8, 16}
    M = 64, N = 8/16/32, K = 64
    M = 128/256/512/1024, N = 8/16/32/64, K = 128/256/512/1024
"""

# datasetEqMNKrptGenSave(1000, 32, 8, 32, [8, 8, 16], [9, 11, 5], 1)
# datasetEqMNKrptGenSave(1000, 32, 8, 32, [16, 16], [9, 5], 2)
# datasetEqMNKrptGenSave(1000, 64, 8, 64, [16, 20, 28], [8, 5, 12], 1)
# datasetEqMNKrptGenSave(1000, 64, 16, 64, [16, 20, 28], [8, 5, 12], 1)
# datasetEqMNKrptGenSave(1000, 64, 32, 64, [16, 20, 28], [8, 5, 12], 1)
# datasetEqMNKrptGenSave(1000, 128, 8, 128, [24, 24, 28, 52], [9, 10, 16, 23], 1)
# datasetEqMNKrptGenSave(1000, 128, 16, 128, [24, 24, 28, 52], [9, 10, 16, 23], 1)
# datasetEqMNKrptGenSave(1000, 128, 32, 128, [24, 24, 28, 52], [9, 10, 16, 23], 1)
datasetEqMNKrptGenSave(100, 1024, 256, 1024, [512, 512], [10, 12], 9)
