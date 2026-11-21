// nvcc inference.cu -o inference -Xcompiler "-O3 -std=c++14" -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -rdc=true
/*
    Check CUDA runtime error:
    cudaError_t err = cudaGetLastError();
    cout << cudaGetErrorString(err) << endl;
*/
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

using namespace std;

void read_mnist_images(const string &path, vector<float> &images)
{
    ifstream file(path, ios::binary);
    if (!file)
    {
        cout << "Cannot open file!" << endl;
        exit(EXIT_FAILURE);
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    file.read((char *)&num_images, sizeof(num_images));
    file.read((char *)&num_rows, sizeof(num_rows));
    file.read((char *)&num_cols, sizeof(num_cols));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
                   ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_images = ((num_images & 0xff000000) >> 24) | ((num_images & 0x00ff0000) >> 8) |
                 ((num_images & 0x0000ff00) << 8) | ((num_images & 0x000000ff) << 24);
    num_rows = ((num_rows & 0xff000000) >> 24) | ((num_rows & 0x00ff0000) >> 8) |
               ((num_rows & 0x0000ff00) << 8) | ((num_rows & 0x000000ff) << 24);
    num_cols = ((num_cols & 0xff000000) >> 24) | ((num_cols & 0x00ff0000) >> 8) |
               ((num_cols & 0x0000ff00) << 8) | ((num_cols & 0x000000ff) << 24);

    int image_size = num_rows * num_cols;
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < image_size; ++j)
        {
            unsigned char pixel = 0;
            file.read((char *)&pixel, sizeof(pixel));
            images.push_back((static_cast<float>(pixel) / 255.0f - 0.5) / 0.5);
        }
    }
}

void read_mnist_labels(string path, vector<int> &labels)
{
    ifstream file(path, ios::binary);
    if (!file)
    {
        cout << "Cannot open file!" << endl;
        exit(EXIT_FAILURE);
    }

    int magic_number = 0, num_items = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    file.read((char *)&num_items, sizeof(num_items));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
                   ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_items = ((num_items & 0xff000000) >> 24) | ((num_items & 0x00ff0000) >> 8) |
                ((num_items & 0x0000ff00) << 8) | ((num_items & 0x000000ff) << 24);

    for (int i = 0; i < num_items; ++i)
    {
        unsigned char label = 0;
        file.read((char *)&label, sizeof(label));
        labels.push_back(static_cast<int>(label));
    }
}

void read_param(string path, vector<float> &params)
{
    ifstream file(path);
    if (!file)
    {
        cout << "Cannot open file!" << endl;
        exit(EXIT_FAILURE);
    }
    string buff;
    while (!file.eof())
    {
        getline(file, buff);
        stringstream curr(buff);
        float param;
        while (curr >> param)
            params.push_back(param);
    }
}

// 1'st Convolutional Layer
__global__ void Conv1(float *input, float *weight, float *bias, float *output)
{
    // each block is responsible for the convolution calculation of one image

    // bx is the image index, by is the channel index
    int bx = blockIdx.x, by = blockIdx.y;
    // ty, tx are indices of row, column in the matrix immediately fetched after the convolution
    int ty = threadIdx.y, tx = threadIdx.x;
    int idx = ty * 24 + tx;

    // weight mask and bias mask in one channel of convolution
    __shared__ float mask[26];
    // the matrix immediately fetched after the convolution
    __shared__ float cvres[576];

    // parallel set of mask
    if (idx < 25)
        mask[idx] = weight[by * 25 + idx];
    if (idx == 25)
        mask[25] = bias[by];
    __syncthreads();

    // parallel calculation of convolution in one channel
    float aml = 0;
    for (int i = ty; i < ty + 5; i++)
        for (int j = tx; j < tx + 5; j++)
            aml += input[bx * 784 + i * 28 + j] * mask[(i - ty) * 5 + (j - tx)];
    aml += mask[25];
    // ReLU activation
    cvres[ty * 24 + tx] = aml * (aml > 0 ? 1.0 : 0.0);
    __syncthreads();

    // max pooling
    float res = cvres[ty * 24 + tx];
    float v2 = cvres[ty * 24 + tx + 1];
    float v3 = cvres[ty * 24 + 24 + tx];
    float v4 = cvres[ty * 24 + 25 + tx];
    if (v2 > res)
        res = v2;
    if (v3 > res)
        res = v3;
    if (v4 > res)
        res = v4;
    if (ty % 2 == 0 && tx % 2 == 0)
        output[bx * 864 + by * 144 + ty * 6 + tx / 2] = res;
}

// 2'nd convolutional layer
__global__ void Conv2(float *input, float *weight, float *bias, float *output)
{
    // bx is the image index, by is the output channel index
    int bx = blockIdx.x, by = blockIdx.y;
    /* ty, tx are indices of row, column in the matrix immediately fetched after the convolution,
       tz is the input channel index */
    int ty = threadIdx.y, tx = threadIdx.x, tz = threadIdx.z;

    int pass = (tz == 0) ? 1 : 0;
    /* weight mask and bias mask in one output channel of convolution, containing weight matrices in
       all input channels */
    __shared__ float mask[151];
    // icres records all the convolution results in all input channels
    __shared__ float icres[384];
    // cvres utilized for accumulation of results from all responding input channels
    __shared__ float cvres[64];

    // parallel set of masks and zero-set of cvres
    if (tx < 5 && ty < 5)
        mask[tz * 25 + ty * 5 + tx] = weight[by * 150 + tz * 25 + ty * 5 + tx];
    if ((ty == 5 && tx == 0) && pass)
        mask[150] = bias[by];
    __syncthreads();

    // accumulation of convolution result of input in all input channels
    icres[ty * 48 + tx * 6 + tz] = 0;
    for (int i = ty; i < ty + 5; i++)
        for (int j = tx; j < tx + 5; j++)
            icres[ty * 48 + tx * 6 + tz] += input[bx * 864 + tz * 144 + i * 12 + j] * mask[tz * 25 + (i - ty) * 5 + (j - tx)];
    __syncthreads();

    // 3-dim block degrades to 2-dim block
    if (pass)
    {
        float pval = mask[150];
        /*
            Upset serial accumulation
            because of lack of understanding to control the hehavior of z-dim threads
        */
        for (int i = 0; i < 6; i++)
        {
            pval += icres[ty * 48 + tx * 6 + i];
        }
        cvres[ty * 8 + tx] = pval * (pval > 0 ? 1.0 : 0.0);
    }
    __syncthreads();
    if (pass)
    {
        float res = cvres[ty * 8 + tx];
        float v2 = cvres[ty * 8 + tx + 1];
        float v3 = cvres[ty * 8 + 8 + tx];
        float v4 = cvres[ty * 8 + 9 + tx];
        if (v2 > res)
            res = v2;
        if (v3 > res)
            res = v3;
        if (v4 > res)
            res = v4;
        if (ty % 2 == 0 && tx % 2 == 0)
            output[bx * 256 + by * 16 + ty * 2 + tx / 2] = res;
    }
}

// 1'st full connection layer
__global__ void FC1(float *input, float *weight, float *bias, float *output)
{
    // each block is responsible for the calculation of one elements in output

    // bx, by is the row, col to calculate
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int idx = ty * 32 + tx;

    float res = 0;

    // utilize all the threads to set the shared memory for the multiplications to obtain 1 element
    __shared__ float As[256];    // 1 rows, 256 cols, row-major
    __shared__ float Bs[256];    // 1 cols, 256 rows, col-major
    __shared__ float mmres[256]; // store the multiplication results

    // !!! Note: B should be the transposition of weight because of the setting in torch.nn.Linear

    if (ty < 8)
        As[idx] = input[bx * 256 + idx];
    else
        Bs[idx - 256] = weight[by * 256 + idx - 256];
    __syncthreads();

    // parallel multiplication
    if (idx < 256)
        mmres[idx] = As[idx] * Bs[idx];
    __syncthreads();

    // accumulation of the multiplication results
    if (idx == 0)
    {
        for (int i = 0; i < 256; i++)
            res += mmres[i];
        res += bias[by];
        // ReLU activation
        output[bx * 120 + by] = res * (res > 0 ? 1.0 : 0.0);
    }
}

// 2'nd full connection layer
__global__ void FC2(float *input, float *weight, float *bias, float *output)
{
    // each block is responsible for the calculation of 2-rows * 2-cols elements in output
    // bx * 2, by * 2 are the 1'st row, 1'st col to calculate
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int idx = ty * 24 + tx;

    float res = 0.0;

    // utilize the threads to set the shared memory for the calculation of 4 elements
    __shared__ float As[240];    // 2 rows, 120 cols, row-major
    __shared__ float Bs[240];    // 2 cols, 120 rows, col-major
    __shared__ float mmres[480]; // store the multiplication results, row major

    // !!! Note: B should be the transposition of weight because of the setting in torch.nn.Linear

    if (ty < 10)
    {
        /*
            bx * 2 * 120 stands for the number of passed vals in A
            idx: 0-->239, coincidentally constitutes a 2-rows space in A
        */
        As[idx] = input[bx * 240 + idx];
    }
    else
    {
        /*
            idx: 240-->359, 1'st col in B involved in the calculation, B_col_id: by + 2,
            weight_row_id: by + 2
            idx: 360-->479, 2'nd col in B involved in the calculation, B_col_id: by + 2 + 1,
            weight_row_id: by + 2 + 1
            idx: 240-->359, B_row_id: 0-->119, weight_col_id: 0-->119
            idx: 360-->479, B_row_id: 0-->119, weight_col_id: 0-->119
        */
        Bs[idx - 240] = weight[by * 240 + idx - 240];
    }
    __syncthreads();

    /*
        expected version:
        The relation between elements to be calculated and idxes:
        idx: 0-->479, element_id: 0-->3
            to calculate res[idx/120], shared memory below are needed
            As:
            element_id: 0, idx: 0-->119, 1'st row of A needed, As_idx: 0-->119
            element_id: 1, idx: 120-->239, 1'st row of A needed, As_idx: 0-->119
            element_id: 2, idx: 240-->359, 2'nd row of A needed, As_idx: 120-->239
            element_id: 3, idx: 360-->479, 2'nd row of A needed, As_idx: 120-->239
            Bs:
            element_id: 0, idx: 0-->119, 1'st col of B needed, Bs_idx: 0-->119
            element_id: 1, idx: 120-->239, 2'nd row of B needed, Bs_idx: 120-->239
            element_id: 2, idx: 240-->359, 1'st col of B needed, Bs_idx: 0-->119
            element_id: 3, idx: 360-->479, 2'nd row of B needed, Bs_idx: 120-->239
        However, the accumulation is not supported in a parallel scenario.
        Therefore, to acquire the multiplication results calculated by each thread,
            shared memory is necessarily introduced.
    */
    mmres[idx] = As[idx - idx / 120 * 120 + idx / 240 * 120] * Bs[idx - idx / 240 * 240];
    __syncthreads();

    if (idx < 4)
    {
        for (int i = 0; i < 120; i++)
            res += mmres[idx * 120 + i];
        res += bias[by * 2 + idx % 2];
        output[bx * 168 + idx / 2 * 84 + by * 2 + idx % 2] = res * (res > 0 ? 1.0 : 0.0);
    }
}

// 3'rd full connection layer
__global__ void FC3(float *input, float *weight, float *bias, float *output)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int idx = ty * 21 + tx;

    float res = 0.0;

    __shared__ float As[168];
    __shared__ float Bs[168];
    __shared__ float mmres[336];

    if (ty < 8)
        As[idx] = input[bx * 168 + idx];
    else
        Bs[idx - 168] = weight[by * 168 + idx - 168];
    __syncthreads();

    mmres[idx] = As[idx - idx / 84 * 84 + idx / 168 * 84] * Bs[idx - idx / 168 * 168];
    __syncthreads();

    if (idx < 4)
    {
        for (int i = 0; i < 84; i++)
            res += mmres[idx * 84 + i];
        res += bias[by * 2 + idx % 2];
        // !!! 3'rd full connection layer shouldn't have a ReLU activation
        output[bx * 20 + idx / 2 * 10 + by * 2 + idx % 2] = res;
    }
}

// classification layer
__global__ void Clf(float *logits, int *labels, int *torf)
{
    // bx is the image index, tx is the column
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int outnum = 0;

    // logits are elements in one row
    __shared__ float logit[10];
    // cmp records all the situation of one-to-one comparison of elements in logits
    __shared__ int cmp[100];
    if (tx < 10)
        logit[tx] = logits[bx * 10 + tx];
    __syncthreads();
    if (logit[tx / 10] > logit[tx - tx / 10 * 10])
        cmp[tx] = 1;
    else
        cmp[tx] = 0;
    __syncthreads();

    if (tx < 10)
    {
        for (int i = 0; i < 10; i++)
            outnum += cmp[tx * 10 + i];
        if (outnum == 9)
        {
            if (tx == labels[bx])
                torf[bx] = 1;
            else
                torf[bx] = 0;
        }
    }
}

__global__ void AccComput(int *torf, float *accuracy)
{
    int correct = 0;
    for (int i = 0; i < 10000; i++)
        if (torf[i] == 1)
            correct++;
    accuracy[0] = correct * 1.0f / 10000;
}

int main(int argc, char *argv[])
{
    string dir = argv[1];

    // Read the images
    vector<float> images;
    read_mnist_images(dir + "/../../data/FashionMNIST/raw/t10k-images-idx3-ubyte", images);
    float *h_images = images.data();
    images.clear();

    // Read the labels
    vector<int> labels;
    read_mnist_labels(dir + "/../../data/FashionMNIST/raw/t10k-labels-idx1-ubyte", labels);
    int *h_labels = labels.data();
    labels.clear();

    // Read the parameters of the model
    vector<float> conv1_w, conv1_b, conv2_w, conv2_b;
    vector<float> fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b;
    float *h_conv1_w, *h_conv1_b, *h_conv2_w, *h_conv2_b;
    float *h_fc1_w, *h_fc1_b, *h_fc2_w, *h_fc2_b, *h_fc3_w, *h_fc3_b;
    read_param(dir + "/conv1.weight.txt", conv1_w);
    read_param(dir + "/conv1.bias.txt", conv1_b);
    read_param(dir + "/conv2.weight.txt", conv2_w);
    read_param(dir + "/conv2.bias.txt", conv2_b);
    read_param(dir + "/fc1.weight.txt", fc1_w);
    read_param(dir + "/fc1.bias.txt", fc1_b);
    read_param(dir + "/fc2.weight.txt", fc2_w);
    read_param(dir + "/fc2.bias.txt", fc2_b);
    read_param(dir + "/fc3.weight.txt", fc3_w);
    read_param(dir + "/fc3.bias.txt", fc3_b);
    h_conv1_w = conv1_w.data();
    h_conv1_b = conv1_b.data();
    h_conv2_w = conv2_w.data();
    h_conv2_b = conv2_b.data();
    h_fc1_w = fc1_w.data();
    h_fc1_b = fc1_b.data();
    h_fc2_w = fc2_w.data();
    h_fc2_b = fc2_b.data();
    h_fc3_w = fc3_w.data();
    h_fc3_b = fc3_b.data();
    conv1_w.clear();
    conv1_b.clear();
    conv2_w.clear();
    conv2_b.clear();
    fc1_w.clear();
    fc1_b.clear();
    fc2_w.clear();
    fc2_b.clear();
    fc3_w.clear();
    fc3_b.clear();

    // Copy data from host to device

    float *d_images;
    cudaMalloc((void **)&d_images, sizeof(float) * 10000 * 28 * 28);
    cudaMemcpy(d_images, h_images, sizeof(float) * 10000 * 28 * 28, cudaMemcpyHostToDevice);

    float *d_conv1_w, *d_conv1_b, *d_conv1_out;
    cudaMalloc((void **)&d_conv1_w, sizeof(float) * 6 * 5 * 5);
    cudaMalloc((void **)&d_conv1_b, sizeof(float) * 6);
    cudaMemcpy(d_conv1_w, h_conv1_w, sizeof(float) * 6 * 5 * 5, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_b, h_conv1_b, sizeof(float) * 6, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_conv1_out, sizeof(float) * 10000 * 6 * 12 * 12);

    float *d_conv2_w, *d_conv2_b, *d_conv2_out;
    cudaMalloc((void **)&d_conv2_w, sizeof(float) * 16 * 6 * 5 * 5);
    cudaMalloc((void **)&d_conv2_b, sizeof(float) * 16);
    cudaMemcpy(d_conv2_w, h_conv2_w, sizeof(float) * 16 * 6 * 5 * 5, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_b, h_conv2_b, sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_conv2_out, sizeof(float) * 10000 * 16 * 4 * 4);

    float *d_fc1_w, *d_fc1_b, *d_fc1_out;
    cudaMalloc((void **)&d_fc1_w, sizeof(float) * 120 * 256);
    cudaMalloc((void **)&d_fc1_b, sizeof(float) * 120);
    cudaMemcpy(d_fc1_w, h_fc1_w, sizeof(float) * 120 * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_b, h_fc1_b, sizeof(float) * 120, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_fc1_out, sizeof(float) * 10000 * 120);

    float *d_fc2_w, *d_fc2_b, *d_fc2_out;
    cudaMalloc((void **)&d_fc2_w, sizeof(float) * 84 * 120);
    cudaMalloc((void **)&d_fc2_b, sizeof(float) * 84);
    cudaMemcpy(d_fc2_w, h_fc2_w, sizeof(float) * 84 * 120, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_b, h_fc2_b, sizeof(float) * 84, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_fc2_out, sizeof(float) * 10000 * 84);

    float *d_fc3_w, *d_fc3_b, *d_fc3_out;
    cudaMalloc((void **)&d_fc3_w, sizeof(float) * 10 * 84);
    cudaMalloc((void **)&d_fc3_b, sizeof(float) * 10);
    cudaMemcpy(d_fc3_w, h_fc3_w, sizeof(float) * 10 * 84, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc3_b, h_fc3_b, sizeof(int) * 10, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_fc3_out, sizeof(float) * 10000 * 10);

    int *d_labels;
    cudaMalloc((void **)&d_labels, sizeof(int) * 10000);
    cudaMemcpy(d_labels, h_labels, sizeof(int) * 10000, cudaMemcpyHostToDevice);
    int *d_torf;
    cudaMalloc((void **)&d_torf, sizeof(int) * 10000 * 1);

    float *h_accuracy = (float *)malloc(sizeof(float)), *d_accuracy;
    cudaMalloc((void **)&d_accuracy, sizeof(float));

    // Set the sizes of the grid and block for each layer
    dim3 conv1_grid(10000, 6);
    dim3 conv1_block(24, 24);
    dim3 conv2_grid(10000, 16);
    dim3 conv2_block(8, 8, 6);
    dim3 fc1_grid(10000, 120);
    dim3 fc1_block(32, 16);
    dim3 fc2_grid(5000, 42);
    dim3 fc2_block(24, 20);
    dim3 fc3_grid(5000, 5);
    dim3 fc3_block(21, 16);
    dim3 clf_grid(10000);
    dim3 clf_block(100);
    dim3 acccompt_grid(1);
    dim3 acccompt_block(1);
    // Using chrono to time
    auto start = chrono::high_resolution_clock::now();

    Conv1<<<conv1_grid, conv1_block>>>(d_images, d_conv1_w, d_conv1_b, d_conv1_out);
    cudaDeviceSynchronize();
    Conv2<<<conv2_grid, conv2_block>>>(d_conv1_out, d_conv2_w, d_conv2_b, d_conv2_out);
    cudaDeviceSynchronize();
    FC1<<<fc1_grid, fc1_block>>>(d_conv2_out, d_fc1_w, d_fc1_b, d_fc1_out);
    cudaDeviceSynchronize();
    FC2<<<fc2_grid, fc2_block>>>(d_fc1_out, d_fc2_w, d_fc2_b, d_fc2_out);
    cudaDeviceSynchronize();
    FC3<<<fc3_grid, fc3_block>>>(d_fc2_out, d_fc3_w, d_fc3_b, d_fc3_out);
    cudaDeviceSynchronize();
    Clf<<<clf_grid, clf_block>>>(d_fc3_out, d_labels, d_torf);
    cudaDeviceSynchronize();
    AccComput<<<acccompt_block, acccompt_grid>>>(d_torf, d_accuracy);
    cudaDeviceSynchronize();
    cudaMemcpy(h_accuracy, d_accuracy, sizeof(float), cudaMemcpyDeviceToHost);

    // End timing
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;

    // Report the result
    cout << fixed << setprecision(4) << diff.count() << ":" << h_accuracy[0] << endl;

    return 0;
}