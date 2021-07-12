#include <omp.h>
#include <iostream>
#include <algorithm>
#include <mkl.h>
#include <immintrin.h>
#include <algorithm>
using namespace std;
constexpr int N = 21250;//4224;//3456;
void mult(const float* A, const float* B, float* C)
{
#pragma omp parallel for
    for (int i = 0; i < N; i++) 
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}
const int bs_c = 48;// 48
const int bs_r = 32;// 32
void mult_block1(const float *A, const float *B, float *C)
{
#pragma omp parallel for
        for (int ib = 0; ib < N / bs_r; ib++)
            for (int kb = 0; kb < N / bs_c; kb++)
                for (int jb = 0; jb < N / bs_c; jb++)
                {
                    const int st = jb * bs_c;
                    for (int i = ib * bs_r; i < ib * bs_r + bs_r; i++)
                    {
                        float* lc = C + i * N + st;
                        for (int k = kb * bs_c; k < kb * bs_c + bs_c; k++)
                        {
                            const float AA = A[i * N + k];
                            const float* lb = B + k * N + st;
                            #pragma simd
                            for (int j = 0; j < bs_c; j++)
                                lc[j] += AA * lb[j];
                        }
                    }
                }
}
void mult_block2(const float* A, const float* B, float* C)
{
#pragma omp parallel
{
        float tmp[bs_c] = { 0.0f };
        #pragma omp for
        for (int ib = 0; ib < N / bs_r; ib++)
            for (int kb = 0; kb < N / bs_c; kb++)
                for (int jb = 0; jb < N / bs_c; jb++)
                {
                    const int st = jb * bs_c;
                    for (int i = ib * bs_r; i < ib * bs_r + bs_r; i++)
                    {
                        for (int k = kb * bs_c; k < kb * bs_c + bs_c; k++)
                        {
                            const float AA = A[i * N + k];
                            const float* pb = B + k * N + st;
                            #pragma simd
                            for (int j = 0; j < bs_c; j++)
                            {
                                tmp[j] += AA * pb[j];
                            }
                        }
                        float* lc = C + i * N + st;
                        #pragma ivdep
                        for (int j = 0; j < bs_c; j++)
                        {
                            lc[j] += tmp[j];
                            tmp[j] = 0.0f;
                        }
                    }
                }
    }
}

void mkl_mult(const float* A, const float* B, float* C)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.f, A, N, B, N, 0.f, C, N);
}

int main()
{
    float* a_host = new float[N * N];
    float* b_host = new float[N * N];
    float* c_host = new float[N * N];
    float* res = new float[N * N];
#pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) a_host[i*N + j] = 1.f/(1.f + i + j);

#pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) b_host[i*N + j] = 9.f / (1.f + i + j);

    std::fill(c_host, c_host+N*N, 0.0f);
    std::fill(res, res+N*N, 0.0f);

    auto time_omp = clock();
    //mult(a_host, b_host, c_host); // mult
    mkl_mult(a_host, b_host, c_host); // mult
    //mult_block1(a_host, b_host, c_host);
    time_omp = clock() - time_omp;
    cout << "Time check mult: " << time_omp << "ms\n";

    time_omp = clock();
    //mult_block2(a_host, b_host, res);//64 32; 64 64 mkl_mult mult_block2
    //mkl_mult(a_host, b_host, res);
    time_omp = clock() - time_omp;
    cout << "Time block mult: " << time_omp << "ms\n";
    //int count = 0;
    //for (int i = 0; i < N*N; i++)
    //{
    //    if (fabs(res[i] - c_host[i]) > 1e-5f)
    //    {
    //        if (count < 15) std::cout << "error i="<< i/N <<" j=" << i%N << " " << res[i] << " " << c_host[i] << "\n";
    //        count++;
    //    }
    //}
    //if (count > 0)
    //{
    //    std::cout << "errors: " << count;
    //    return 0;
    //}
    //std::cout << "ok";
    return 0;
}
//float* allocMatrix(int n)
//{
//    float* mat = (float*)_mm_malloc(sizeof(float) * (n * n), 64);
//    return mat;
//}