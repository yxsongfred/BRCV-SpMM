/*
    Utils for batched SpMMs
    coded and annotated by Yuxuan SONG
    Last modified time: May 26, 2024
*/
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <cmath>

using namespace std;

void datasetMap(int argc, char *argv[], string &foldername)
{
    string pfx = "./data/eqMNKrpt/";
    switch (atoi(argv[1]))
    {
    case 0:
        foldername = pfx + "M32_N8_K32/1/";
        break;
    default:
        exit(EXIT_FAILURE);
    }
}

/*
    Retrieve int data from a csv-format file
*/
void readcsv(string filepath, vector<int> &data)
{
    ifstream file(filepath);
    if (!file.is_open())
    {
        cout << "Error when attempting to open file: " << filepath << "!" << endl;
        exit(EXIT_FAILURE);
    }
    string buff;
    while (!file.eof())
    {
        getline(file, buff, ',');
        stringstream curr(buff);
        int value;
        while (curr >> value)
            data.push_back(value);
    }
}
/*
    Retrieve float data from a csv-format file
*/
void readcsv(string filepath, vector<float> &data)
{
    ifstream file(filepath);
    if (!file.is_open())
    {
        cout << "Error when attempting to open file: " << filepath << "!" << endl;
        exit(EXIT_FAILURE);
    }
    string buff;
    while (!file.eof())
    {
        getline(file, buff, ',');
        stringstream curr(buff);
        float value;
        while (curr >> value)
            data.push_back(value);
    }
}

/*
    Append int values in the source vector to the destination int vector
*/
void vecAppend(vector<int> src, vector<int> &dest)
{
    for (auto &e : src)
        dest.push_back(move(e));
}
/*
    Append float values in the source vector to the destination float vector
*/
void vecAppend(vector<float> src, vector<float> &dest)
{
    for (auto &e : src)
        dest.push_back(move(e));
}

/*
    CSR format:
        Premise: a SpM with a dimension of M * K
        Abbreviation illustration:
            nnz stands for the number of non-zero elements
                (unless otherwise specified, nnz is counted in a whole SpM)
            rpt[M + 1]:
                row pointer
                rpt[i] indicates the sum of nnz's in the first i'th rows
            cid[nnz]:
                column IDs of non-zero elements
            val[nnz]:
                values of non-zero elements
    ----------------
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

/*
    A naiive conversion from a CSR-format SpM into a BRCV-format SpM
    In the same sparse band,
    all the elements in any column with at least one non-zero element are treated as non-zero elements.
    Array b is manually set
*/
void csr2brcvNaiive(vector<int> rpt, vector<int> cid, vector<float> val,
                    vector<int> &b, vector<int> &rc, vector<int> &rv, vector<int> &c, vector<float> &v)
{
    if (b.size() == 0)
    {
        cout << "Undefined heights of sparse bands!" << endl;
        exit(EXIT_FAILURE);
    }
    int p = b.size() - 1;
    rc.push_back(0);
    rv.push_back(0);
    int i, j, k, l;
    vector<int> trc;
    vector<int>::iterator trcend;
    vector<float> ttv, tv;

    for (i = 1; i < p + 1; i++)
    // Traverse each sparse band (1'st ~ p'th)
    {
        // Obtain the union of sets of column IDs within H rows in a sparse band
        for (j = b[i - 1]; j < b[i]; j++)
            // Traverse each row in the i'th sparse band
            for (k = rpt[j]; k < rpt[j + 1]; k++)
                // Traverse each non-zero element in j'th row
                trc.push_back(cid[k]);
        sort(trc.begin(), trc.end());
        // Obtain the union set
        trcend = unique(trc.begin(), trc.end());
        trc.erase(trcend, trc.end());
        // Set values in arrays rc, rv, c
        rc.push_back(rc[i - 1] + trc.size());
        if (i < p)
            rv.push_back(rv[i - 1] + (b[i] - b[i - 1]) * trc.size());
        vecAppend(trc, c);
        // Store the non-zero elements in BRCV format
        for (j = b[i - 1]; j < b[i]; j++)
        {
            l = 0;
            ttv.assign(trc.size(), 0);
            // Zero paddings
            for (k = rpt[j]; k < rpt[j + 1]; k++)
            {
                for (; l < trc.size(); l++)
                {
                    if (cid[k] == trc[l])
                    {
                        ttv[l] = val[k];
                        break;
                    }
                }
            }
            vecAppend(ttv, tv);
            ttv.clear();
        }
        trc.clear();
        // Set values in array v
        vecAppend(tv, v);
        tv.clear();
    }
}

/*
    Conversion from CSR-format SpM to Dense-format "DeM"
*/
void csr2dense(vector<int> rpt, vector<int> cid, vector<float> val, int ncol, vector<float> &dem)
{
    int i, j, k;
    vector<float> trow;
    for (i = 0; i < rpt.size() - 1; i++)
    {
        k = 0;
        trow.assign(ncol, 0);
        // Zero paddings
        for (j = rpt[i]; j < rpt[i + 1]; j++)
        {
            for (; k < ncol; k++)
            {
                if (cid[j] == k)
                {
                    trow[k] = val[j];
                    break;
                }
            }
        }
        vecAppend(trow, dem);
        trow.clear();
    }
}
/*
    Conversion from CSR-format SpM to Dense-format "DeM" (2-D representation)
*/
void csr2dense(vector<int> rpt, vector<int> cid, vector<float> val, int ncol, vector<vector<float>> &dem)
{
    int i, j, k;
    vector<float> trow;
    for (i = 0; i < rpt.size() - 1; i++)
    {
        k = 0;
        trow.assign(ncol, 0);
        // Zero paddings
        for (j = rpt[i]; j < rpt[i + 1]; j++)
        {
            for (; k < ncol; k++)
            {
                if (cid[j] == k)
                {
                    trow[k] = val[j];
                    break;
                }
            }
        }
        dem.push_back(trow);
        trow.clear();
    }
}

/*
    Transpose a matrix
        om: original matrix
        tm: transposed matrix of the original matrix
        ldom: leading dimension (#cols) of om
            ldX means the leading dimension of matrix X
            ldX = #cols in X iff X is row-major stored
            ldX = #rows in X iff X is column-major stored
*/
void mtxTranspose(vector<float> om, vector<float> &tm, int ldom)
{
    // ldtm: leading dimension of tm
    int ldtm = om.size() / ldom;
    tm.assign(ldtm * ldom, 0);
    for (int i = 0; i < ldom; i++)
        for (int j = 0; j < ldtm; j++)
            tm[j * ldom + i] = om[i * ldtm + j];
}

/*
    Find the minimun factor of N whose value is greater or equal to K
*/
int minFactorNgeK(int N, int K)
{
    if (K > N)
    {
        cout << "K must be less than N!" << endl;
        exit(EXIT_FAILURE);
    }
    vector<int> lf; // Factors of N which are greater than the square root of N
    lf.push_back(N);
    int i, res = 0, tmp;
    for (i = 2; i <= int(sqrt(N)); i++)
    // Traverse to obtain factors of N
    {
        if (N % i == 0) // A factor of N is encountered
        {
            if (i >= K)
            {
                res = i; // If minFactorNgeK is found, return res
                break;
            }
            else
                lf.push_back(N / i); // Elements in lf is arranged in descending order
        }
    }
    if (!res) // If minFactorNgeK is greater than the square root of N
    {
        for (i = lf.size() - 1; i >= 0; i--) // Traverse lf from tail to head
        {
            tmp = lf[i];
            if (tmp >= K) // If minFactorNgeK is found, return res
            {
                res = tmp;
                break;
            }
        }
    }
    return res;
}

/*
    Verification of batched MM result (matrices stored in 1-D array)
    expt_res: result obtained by batched matrix-matrix algorithms
    teo_res: theoretical result of batched MM (stored beforehead)
*/
void verify(float *expt_res, vector<float> teo_res)
{
    for (int i = 0; i < teo_res.size(); i++)
        if (fabs(expt_res[i] - teo_res[i]) > 1e-2)
        {
            cout << "Wrong answer starts from the " << i << "'th place!" << endl;
            exit(EXIT_FAILURE);
        }
}
/*
    Verification of batched MM result (matrices stored in 2-D array)
    expt_res: result obtained by batched matrix-matrix algorithms
    teo_res: theoretical result of batched MM (stored beforehead)
*/
void verify(vector<vector<float>> expt_res, vector<vector<float>> teo_res)
{
    for (int i = 0; i < teo_res.size(); i++)
    {
        for (int j = 0; j < teo_res[i].size(); j++)
        {
            if (fabs(expt_res[i][j] - teo_res[i][j]) > 1e-2)
            {
                cout << "Wrong answer starts from "
                     << "the " << j << "'th place in the " << i << "'th matrix" << endl;
                exit(EXIT_FAILURE);
            }
        }
    }
}