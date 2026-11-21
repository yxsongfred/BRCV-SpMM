# Efficient batched multiplication of band sparse matrix and dense matrix on GPU
This is a reproduction of a Master's thesis available online in [website](http://dpaper.las.ac.cn/Dpaper/detail/detailNew?paperID=20217376&title=%E5%B8%A6%E7%8A%B6%E7%A8%80%E7%96%8F%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%E7%9A%84%E9%AB%98%E6%95%88GPU%E5%AE%9E%E7%8E%B0%E5%8F%8A%E5%BA%94%E7%94%A8&author=%E5%88%98%E4%B8%BD&highsearch=author_name%253A345345345%25E5%2588%2598%25E4%25B8%25BD345345345%2520%2520234234234234234234%2520%2520specification_training_institution_str%253A%25E4%25B8%25AD%25E5%259B%25BD%25E7%25A7%2591%25E5%25AD%25A6%25E9%2599%25A2%25E9%2587%258D%25E5%25BA%2586%25E7%25BB%25BF%25E8%2589%25B2%25E6%2599%25BA%25E8%2583%25BD%25E6%258A%2580%25E6%259C%25AF%25E7%25A0%2594%25E7%25A9%25B6%25E9%2599%25A2&sortField=score%2520desc%252Cid&start=0&actionType=Browse&searchText=%E5%88%98%E4%B8%BD)
## Programming language
CUDA, Python
## File organization
- [`gemmBatched.cu`](https://github.com/yxsongfred/BRCV-SpMM/blob/main/gemmBatched.cu):
Batched matrix-matrix multiplication (GEMM) accelerated by cuBLAS
- [`gemmSerial.cu`](https://github.com/yxsongfred/BRCV-SpMM/blob/main/gemmBatched.cu):
Serial implementation of batched matrix-matrix multiplication accelerated by cuBLAS
- [`spmmCSRBatched.cu`](https://github.com/yxsongfred/BRCV-SpMM/blob/main/spmmCSRBatched.cu):
Batched CSR-format sparse matrix-matrix multiplication (SpMM) accelerated by cuSPARSE
- [`spmmCSRSerial.cu`](https://github.com/yxsongfred/BRCV-SpMM/blob/main/spmmCSRSerial.cu):
Serial implementation of CSR-format sparse matrix-matrix multiplication accelerated by cuSPARSE
- [`spmmBRCVBatched_v1.cu`](https://github.com/yxsongfred/BRCV-SpMM/blob/main/spmmBRCVBatched_v1.cu)
Batched BRCV-format sparse matrix - matrix multiplication (Main Entry)
- [`spmmBRCVBatched_v1.cuh`](https://github.com/yxsongfred/BRCV-SpMM/blob/main/spmmBRCVBatched_v1.cuh): Batched BRCV-format sparse matrix - matrix multiplication (GPU kernel)
- [`utils.h`](https://github.com/yxsongfred/BRCV-SpMM/blob/main/utils.h):
Utils for batched SpMMs
- [`data/`](https://github.com/yxsongfred/BRCV-SpMM/tree/main/data):
Adjacency matrix (sparse) and dense matrix (dense) in real-word applications (Cora as an example)
- [`DatasetGenerate.py`](https://github.com/yxsongfred/BRCV-SpMM/blob/main/DatasetGenerate.py):
Generate experimental datasets
- [`nodeReordering/`](https://github.com/yxsongfred/BRCV-SpMM/tree/main/nodeReordering):
C++ implementation of Reverse Cuthill-Mckee (RCM) algorithm for node reordering in adjacency matrix
- [`gpu_course/`](https://github.com/yxsongfred/BRCV-SpMM/tree/main/gpu_course):
My implementation of homework for UCAS 23 Fall course "GPU Architecture and Programming"