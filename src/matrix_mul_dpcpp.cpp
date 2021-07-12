// https://software.intel.com/en-us/oneapi-programming-guide
#include <omp.h>
#include <algorithm>
#include <CL/sycl.hpp>
#include <iostream>
#include <limits>

#include "dpc_common.hpp"
class gpu_block3;
#include "oneapi/mkl/blas.hpp"
#include <mkl.h>
using namespace std;
using namespace cl::sycl;

constexpr int N = 4032;//3456;//4032
const int bs_c = 32;// 48;
const int bs_r = 32;
const int tile_size = 32; //gpu_mullt_block2 - 16; gpu_mullt_block4 - 32
void mult(const float* A, const float* B, float* C)
{
#pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];

}
void mult_block2(const float* A, const float* B, float* C)
{
    auto time_omp = clock();
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
                            #pragma omp simd // ivdep
                            for (int j = 0; j < bs_c; j++)
                                tmp[j] += AA * pb[j];
                        }
                        float* lc = C + i * N + st;
                        //#pragma omp simd
                        for (int j = 0; j < bs_c; j++)
                        {
                            lc[j] += tmp[j];
                            tmp[j] = 0.0f;
                        }
                    }
                }
    }
    time_omp = clock() - time_omp;
    std::cout << "Time cpu block mult: " << time_omp << "ms\n";
}
void mult_mkl(const float* A, const float* B, float* C)
{
    auto time_omp = clock();
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.f, A, N, B, N, 0.f, C, N);
    oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;
    try 
    {
        queue q(default_selector{}, dpc_common::exception_handler);
        std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;
        
        buffer<float, 1> a{A, range<1>{N*N}};
        buffer<float, 1> b{B, range<1>{N*N}};
        buffer<float, 1> c{C, range<1>{N*N}};
        oneapi::mkl::blas::gemm(q, transA, transB, N, N, N, 1.0f, a, N, b, N, 0.0f, c, N);
        q.wait_and_throw();
    }
    catch (cl::sycl::exception const& e) {
        std::cout << "\t\tSYCL exception during GEMM\n" << e.what() << std::endl << "OpenCL status: " << e.get_cl_code() << std::endl;
    }
    time_omp = clock() - time_omp;
    std::cout << "Time mkl mult: " << time_omp << "ms\n";
}
void gpu_mullt_base(const float* a_host, const float* b_host, float* c_back)
{
    try {
        queue q(gpu_selector{}, dpc_common::exception_handler); //default_selector
        cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
        buffer<float, 2> a_buf(a_host, sycl::range(N, N));
        buffer<float, 2> b_buf(b_host, range(N, N));
        buffer c_buf(reinterpret_cast<float*>(c_back), range(N, N));
        // инициализируем память gpu
        q.submit([&](auto& h) {
            accessor a(a_buf, h, write_only);
            accessor b(b_buf, h, write_only);
            h.single_task([=]() {
                a[0][0] = a[0][0];
                b[0][0] = b[0][0];
                });
            });

        q.wait_and_throw();
        std::cout << "gpu base range 2 start\n";
        auto time_dpc = clock();
        q.submit([&](auto& h) {
            accessor a(a_buf, h, read_only);
            accessor b(b_buf, h, read_only);
            accessor c(c_buf, h, write_only);
            int width_a = a_buf.get_range()[1];

            h.parallel_for(range(N, N), [=](auto index) {
                int row = index[0];
                int col = index[1];
                float sum = 0.0f;
                for (int i = 0; i < width_a; i++)
                    sum += a[row][i] * b[i][col];
                c[index] = sum;
                });
            });
        q.wait_and_throw();
        time_dpc = clock() - time_dpc;
        cout << "Time dpc mult: " << time_dpc << "ms\n";
    }
    catch (cl::sycl::exception const& e) {
        cout << "An exception is caught while multiplying matrices.\n";
        terminate();
    }
}
void gpu_mullt_native(const float* a_host, const float* b_host, float* c_back)
{
    try {
        queue q(default_selector{}, dpc_common::exception_handler); //default_selector
        cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
        buffer<float, 2> a_buf(a_host, range(N, N));
        buffer<float, 2> b_buf(b_host, range(N, N));
        buffer c_buf(reinterpret_cast<float*>(c_back), range(N, N));
        // инициализируем память gpu
        q.submit([&](auto& h) {
            accessor a(a_buf, h, write_only);
            accessor b(b_buf, h, write_only);
            h.single_task([=]() {
                a[0][0] = a[0][0];
                b[0][0] = b[0][0];
                });
            });

        q.wait_and_throw();
        std::cout << "gpu native range 2 start\n";
        auto time_dpc = clock();
        q.submit([&](auto& h) {
            accessor a(a_buf, h, read_only);
            accessor b(b_buf, h, read_only);
            accessor c(c_buf, h, write_only);
            int wA = a_buf.get_range()[1];

            h.parallel_for(range(N, N), [=](auto index) {
                int row = index[0];
                int col = index[1];
                for (int i = 0; i < wA; i++)
                    c[row][col] += a[row][i] * b[i][col];
                });
            });
        q.wait_and_throw();
        time_dpc = clock() - time_dpc;
        cout << "Time dpc mult: " << time_dpc << "ms\n";
    }
    catch (cl::sycl::exception const& e) {
        cout << "An exception is caught while multiplying matrices.\n";
        terminate();
    }
}
void gpu_mullt_block(const float* a_host, const float* b_host, float* c_back);
void gpu_mullt_block2(const float* a_host, const float* b_host, float* c_back)
{
    try
    {
        queue q(default_selector{}, dpc_common::exception_handler);
        std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
        buffer<float, 1> a_buf(a_host, range(N*N));
        buffer<float, 1> b_buf(b_host, range(N*N));
        buffer c_buf(reinterpret_cast<float*>(c_back), range(N*N));

        // инициализируем память до начала вычислений
        q.submit([&](auto& h) {
            accessor a(a_buf, h, write_only);
            accessor b(b_buf, h, write_only);
            h.single_task([=]() {
                a[0] = a[0];
                b[0] = b[0];
                });
            });
        q.wait_and_throw();

        std::cout << "gpu_block2 start\n";
        auto time_dpc = clock();

        q.submit([&](auto& h) {
            accessor a(a_buf, h, read_only);
            accessor b(b_buf, h, read_only);
            accessor c(c_buf, h, write_only);

            accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> aTile(cl::sycl::range<1>(tile_size*tile_size), h);
            accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> bTile(cl::sycl::range<1>(tile_size*tile_size), h);

            range<2> matrix_range{ N, N };
            range<2> tile_range{ tile_size, tile_size };
            h.parallel_for(cl::sycl::nd_range<2>(matrix_range, tile_range), [=](auto it) {
                int row = it.get_local_id(0);
                int col = it.get_local_id(1);
                const int globalRow = tile_size * it.get_group(0) + row;
                const int globalCol = tile_size * it.get_group(1) + col;
                const int numTiles = N / tile_size;
                float sum = 0.0f;
                for (int t = 0; t < numTiles; t++)
                {
                    aTile[row * tile_size + col] = a[globalRow * N + tile_size * t + col];
                    bTile[row * tile_size + col] = b[(tile_size * t + row) * N + globalCol];
                    it.barrier(cl::sycl::access::fence_space::local_space);
                    #pragma unroll(2)
                    for (int k = 0; k < tile_size; k++)
                        sum += aTile[row * tile_size + k] * bTile[k * tile_size + col];
                    it.barrier(cl::sycl::access::fence_space::local_space);
                }
                c[globalRow*N+globalCol] = sum;
                });
            }).wait_and_throw();
            time_dpc = clock() - time_dpc;
            std::cout << "Time dpc mult: " << time_dpc << "ms\n";
    }
    catch (cl::sycl::exception const& e) {
        std::cout << "An exception is caught while multiplying matrices.\n" ;
        terminate();
    }
}
void gpu_mullt_block3(const float* a_host, const float* b_host, float* c_back);
void gpu_mullt_block4(const float* a_host, const float* b_host, float* c_back)
{
    try
    {
        queue q(default_selector{}, dpc_common::exception_handler);
        std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
        buffer<float, 1> a_buf(a_host, range(N * N));
        buffer<float, 1> b_buf(b_host, range(N * N));
        buffer c_buf(reinterpret_cast<float*>(c_back), range(N * N));

        // инициализируем память до начала вычислений
        q.submit([&](auto& h) {
            accessor a(a_buf, h, write_only);
            accessor b(b_buf, h, write_only);
            h.single_task([=]() {
                a[0] = a[0];
                b[0] = b[0];
                });
            });
        q.wait_and_throw();

        std::cout << "gpu_block4 start\n";
        auto time_dpc = clock();
        q.submit([&](auto& h) {
            accessor a(a_buf, h, read_only);
            accessor b(b_buf, h, read_only);
            accessor c(c_buf, h, write_only);

            accessor<float, 1, access::mode::read_write, cl::sycl::access::target::local> aTile(cl::sycl::range<1>(tile_size*tile_size), h);
            accessor<float, 1, access::mode::read_write, cl::sycl::access::target::local> bTile(cl::sycl::range<1>(tile_size*tile_size), h);

            range<2> matrix_range{ N / 4, N };
            range<2> tile_range{ tile_size / 4, tile_size };
            h.parallel_for(cl::sycl::nd_range<2>(matrix_range, tile_range), [=](auto it) {
                int row = it.get_local_id(0);
                int col = it.get_local_id(1);
                const int globalRow = tile_size * it.get_group(0) + row;
                const int globalCol = tile_size * it.get_group(1) + col;
                const int numTiles = N / tile_size;
                float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
                const int tmp = tile_size / 4;
                for (int t = 0; t < numTiles; t++)
                {
                    aTile[row*tile_size+col] = a[globalRow * N + tile_size * t + col];
                    bTile[row*tile_size+col] = b[(tile_size * t + row) * N + globalCol];
                    
                    aTile[(row + tmp)*tile_size+col] = a[globalRow * N + tile_size * t + col + tmp * N];
                    bTile[(row + tmp)*tile_size+col] = b[(tile_size * t + row) * N + globalCol + tmp * N];
                    
                    aTile[(row + 2 * tmp)*tile_size+col] = a[globalRow * N + tile_size * t + col + 2 * tmp * N];
                    bTile[(row + 2 * tmp)*tile_size+col] = b[(tile_size * t + row) * N + globalCol + 2 * tmp * N];
                    
                    aTile[(row + 3 * tmp)*tile_size+col] = a[globalRow * N + tile_size * t + col + 3 * tmp * N];
                    bTile[(row + 3 * tmp)*tile_size+col] = b[(tile_size * t + row) * N + globalCol + 3 * tmp * N];
                    
                    it.barrier(cl::sycl::access::fence_space::local_space);
                    #pragma unroll(2)
                    for (int k = 0; k < tile_size; k++)
                    {
                        sum1 += aTile[row*tile_size+k] * bTile[k*tile_size+col];
                        sum2 += aTile[(row + tmp)*tile_size+k] * bTile[k*tile_size+col];
                        sum3 += aTile[(row + 2 * tmp)*tile_size+k] * bTile[k*tile_size+col];
                        sum4 += aTile[(row + 3 * tmp)*tile_size+k] * bTile[k*tile_size+col];
                    }
                    it.barrier(cl::sycl::access::fence_space::local_space);
                }
                c[globalRow * N + globalCol] = sum1;
                c[(globalRow + tmp) * N + globalCol] = sum2;
                c[(globalRow + 2 * tmp) * N + globalCol] = sum3;
                c[(globalRow + 3 * tmp) * N + globalCol] = sum4;
                });
            }).wait_and_throw();
            time_dpc = clock() - time_dpc;
            std::cout << "Time dpc mult: " << time_dpc << "ms\n";
    }
    catch (cl::sycl::exception const& e) {
        std::cout << "An exception is caught while multiplying matrices.\n";
        terminate();
    }
}
int main()
{
  // Host memory buffer that device will write data back before destruction.
    float* a_host = new float[N * N];
    float* b_host = new float[N * N];
    float* c_host = new float[N * N];
    float* res = new float[N * N];
#pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) a_host[i * N + j] = 1.f / (N + i + j);// (1.f + i + j);

#pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) b_host[i * N + j] = N / (1.f + i + j);

    std::fill(c_host, c_host + N * N, 0.0f);
    std::fill(res, res + N * N, 0.0f);

    //mult(a_host, b_host, c_host);
    mult_block2(a_host, b_host, c_host);
    //mult_mkl(a_host, b_host, c_host);
    //gpu_mullt_block4(a_host, b_host, c_host);

    //gpu_mullt_native(a_host, b_host, res);
    //gpu_mullt_base(a_host, b_host, res);
    //gpu_mullt_block2(a_host, b_host, res);
    gpu_mullt_block4(a_host, b_host, res);

    int count = 0;
    float max_delta = 0.0f;
    for (int i = 0; i < N * N; i++)
    {
        //std::cout << res[i] << " " << c_host[i] << "\n";
        max_delta = fmax(fabs(res[i] - c_host[i]), max_delta);
        if (fabs(res[i] - c_host[i]) > 1e-2)
        {
            if (count < 5) std::cout << "error i=" << i / N << " j=" << i % N << " " << res[i] << " expected " << c_host[i] << "\n";
            count++;
        }
    }
    if (count > 0)
    {
        std::cout << "\n\nerrors: " << count << " max delta: " << max_delta << "\n";
        std::cout << "(2,2) " << res[2*N+2] << " (2, 2) " << c_host[2*N + 2] << " (1,2) " << res[1 * N + 2] << " (2, 1) " << c_host[2 * N + 1];
        return 0;
    }
    std::cout << "ok";
    return 0;
}
void gpu_mullt_block(const float* a_host, const float* b_host, float* c_back)
{
    try
    {
        queue q(default_selector{}, dpc_common::exception_handler);
        std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
        buffer<float, 2> a_buf(a_host, range(N, N));
        buffer<float, 2> b_buf(b_host, range(N, N));
        buffer c_buf(reinterpret_cast<float*>(c_back), range(N, N));

        // инициализируем память до начала вычислений
        q.submit([&](auto& h) {
            accessor a(a_buf, h, write_only);
            accessor b(b_buf, h, write_only);
            h.single_task([=]() {
                a[0][0] = a[0][0];
                b[0][0] = b[0][0];
                });
            });
        q.wait_and_throw();

        std::cout << "gpu_block start\n";
        auto time_dpc = clock();
        // Submit command group to queue to multiply matrices: c = a * b
        q.submit([&](auto& h) {
            // Read from a and b, write to c
            accessor a(a_buf, h, read_only);
            accessor b(b_buf, h, read_only);
            accessor c(c_buf, h, write_only);

            accessor<float, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> aTile(cl::sycl::range<2>(tile_size, tile_size), h);
            accessor<float, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> bTile(cl::sycl::range<2>(tile_size, tile_size), h);

            range<2> matrix_range{ N, N };
            range<2> tile_range{ tile_size, tile_size };
            // Execute kernel.
            h.parallel_for(cl::sycl::nd_range<2>(matrix_range, tile_range), [=](auto ind) {
                int row = ind.get_local_id(0);
                int col = ind.get_local_id(1);
                const int globalRow = tile_size * ind.get_group(0) + row;
                const int globalCol = tile_size * ind.get_group(1) + col;
                const int numTiles = N / tile_size;
                float sum = 0.0f;
                for (int t = 0; t < numTiles; t++)
                {
                    aTile[row][col] = a[globalRow][tile_size * t + col];
                    bTile[row][col] = b[tile_size * t + row][globalCol];
                    ind.barrier(cl::sycl::access::fence_space::local_space);
                    //#pragma unroll 2
                    for (int k = 0; k < tile_size; k++)
                        sum += aTile[row][k] * bTile[k][col];
                    ind.barrier(cl::sycl::access::fence_space::local_space);
                }
                c[globalRow][globalCol] = sum;
                });
            }).wait_and_throw();
            time_dpc = clock() - time_dpc;
            std::cout << "Time dpc mult: " << time_dpc << "ms\n";
    }
    catch (cl::sycl::exception const& e) {
        std::cout << "An exception is caught while multiplying matrices.\n";
        terminate();
    }
}
void gpu_mullt_block3(const float* a_host, const float* b_host, float* c_back)
{
    try
    {
        queue q(default_selector{}, dpc_common::exception_handler);
        std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
        buffer<float, 1> a_buf(a_host, range(N * N));
        buffer<float, 1> b_buf(b_host, range(N * N));
        buffer c_buf(reinterpret_cast<float*>(c_back), range(N * N));

        // инициализируем память до начала вычислений
        q.submit([&](auto& h) {
            accessor a(a_buf, h, write_only);
            accessor b(b_buf, h, write_only);
            h.single_task([=]() {
                a[0] = a[0];
                b[0] = b[0];
                });
            });
        q.wait_and_throw();

        std::cout << "gpu_block3 start\n";
        auto time_dpc = clock();
        q.submit([&](auto& h) {
            accessor a(a_buf, h, read_only);
            accessor b(b_buf, h, read_only);
            accessor c(c_buf, h, write_only);

            accessor<float, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> aTile(cl::sycl::range<2>(tile_size, tile_size), h);
            accessor<float, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> bTile(cl::sycl::range<2>(tile_size, tile_size), h);

            range<2> matrix_range{ N / 4, N };
            range<2> tile_range{ tile_size / 4, tile_size };
            h.parallel_for<gpu_block3>(cl::sycl::nd_range<2>(matrix_range, tile_range), [=](auto it) {
                int row = it.get_local_id(0);
                int col = it.get_local_id(1);
                const int globalRow = tile_size * it.get_group(0) + row;
                const int globalCol = tile_size * it.get_group(1) + col;
                const int numTiles = N / tile_size;
                float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
                const int tmp = tile_size / 4;
                for (int t = 0; t < numTiles; t++)
                {
                    aTile[row][col] = a[globalRow*N+tile_size*t+col];
                    bTile[row][col] = b[(tile_size*t+row)*N+globalCol];

                    aTile[row+tmp][col] = a[globalRow*N+tile_size*t+col+tmp*N];
                    bTile[row+tmp][col] = b[(tile_size*t+row)*N+globalCol+tmp*N];

                    aTile[row+2*tmp][col] = a[globalRow*N+tile_size*t+col+2*tmp*N];
                    bTile[row+2*tmp][col] = b[(tile_size*t+row)*N+globalCol+2*tmp*N];

                    aTile[row+3*tmp][col] = a[globalRow*N+tile_size*t+col+3*tmp*N];
                    bTile[row+3*tmp][col] = b[(tile_size*t+row)*N+globalCol+3*tmp*N];

                    it.barrier(cl::sycl::access::fence_space::local_space);
                    #pragma unroll(8)
                    for (int k = 0; k < tile_size; k++)
                    {
                        sum1 += aTile[row][k] * bTile[k][col];
                        sum2 += aTile[row+tmp][k] * bTile[k][col];
                        sum3 += aTile[row+2*tmp][k] * bTile[k][col];
                        sum4 += aTile[row+3*tmp][k] * bTile[k][col];
                    }
                    it.barrier(cl::sycl::access::fence_space::local_space);
                }
                c[globalRow*N + globalCol] = sum1;
                c[(globalRow+tmp)*N+globalCol] = sum2;
                c[(globalRow+2*tmp)*N+globalCol] = sum3;
                c[(globalRow+3*tmp)*N+globalCol] = sum4;
                });
            }).wait_and_throw();
            time_dpc = clock() - time_dpc;
            std::cout << "Time dpc mult: " << time_dpc << "ms\n";
    }
    catch (cl::sycl::exception const& e) {
        std::cout << "An exception is caught while multiplying matrices.\n";
        terminate();
    }
}
