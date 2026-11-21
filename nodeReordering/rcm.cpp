/*
    Node reordering of sparse matrices via Reverse Cuthill-Mckee (RCM) algorithm
    Ideas inspired by https://www.geeksforgeeks.org/reverse-cuthill-mckee-algorithm/
    Recoded and annotated by Yuxuan SONG
    Last modifed time: Apr 19, 2024
*/
#include <iostream>
#include <vector>
#include "utils.h"
#include <queue>
using namespace std;

// CSR-format sparse matrix

int main()
{
    vector<int> rpt, cid;
    vector<float> val;
    readcsv("./data/Cora.data.csv", rpt);
    readcsv("./data/Cora.indices.csv", cid);
    readcsv("./data/Cora.data.csv", val);
    vector<vector<float>> A_ori;
    csr2dense(rpt, cid, val, rpt.size() - 1, A_ori);
    queue<int> Q;
    vector<int> R;
    vector<int> degree_list;
    for (int i = 1; i < rpt.size(); i++)
        degree_list.push_back(rpt[i] - rpt[i - 1]);

    return 0;
}