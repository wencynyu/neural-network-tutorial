#include <immintrin.h>
#include <ctime>
#include <iostream>

int main(int argc, char const *argv[])
{
    int len = 2100000000;

    // Dynamic memory allocation with 64-byte alignment
    float *pInVector = (float *)_mm_malloc(len * sizeof(float), 64);
    float *pOutVector = (float *)_mm_malloc(len * sizeof(float), 64);

    // init data
    for (int i = 0; i < len; i++)
        pInVector[i] = 1;

    // Static memory allocation of 16 floats with 64-byte alignments
    alignas(64) float vec_1[16] = {1, 1, 1, 1,
                                   1, 1, 1, 1,
                                   1, 1, 1, 1,
                                   1, 1, 1, 1};
    alignas(64) float vec_2[16] = {-1, -1, -1, -1,
                                   -1, -1, -1, -1,
                                   -1, -1, -1, -1,
                                   -1, -1, -1, -1};

    //__m512 data type represents a Zmm register with 16 float elements
    __m512 Zmm_vec_2 = _mm512_load_ps(vec_2);

    // Intel® AVX-512 512-bit packed single load
    __m512 Zmm_vec_1 = _mm512_load_ps(vec_1);
    __m512 Zmm0, Zmm1, Zmm2, Zmm3;

    std::clock_t start = std::clock();
    // 对 pInVector 中的每 32 个元素，计算 (in_val * vec_2) - (in_val * vec_1)，其中 in_val 是从 pInVector 中加载的元素
    for (int i = 0; i < len; i += 32)
    {
        // 512 / 32 = 16
        // 512 / 32 = 16
        Zmm0 = _mm512_load_ps(pInVector + i); // Load 16 floats from pInVector
        Zmm1 = _mm512_mul_ps(Zmm0, Zmm_vec_1); // Zmm1 = in_val * vec_1
        Zmm2 = _mm512_mul_ps(Zmm0, Zmm_vec_2); // Zmm2 = in_val * vec_2
        Zmm3 = _mm512_sub_ps(Zmm1, Zmm2); // Zmm3 = Zmm1 - Zmm2
        _mm512_store_ps(pOutVector + i, Zmm3);

        Zmm0 = _mm512_load_ps(pInVector + i + 16); // Load next 16 floats from pInVector
        Zmm1 = _mm512_mul_ps(Zmm0, Zmm_vec_1); // Zmm1 = in_val * vec_1
        Zmm2 = _mm512_mul_ps(Zmm0, Zmm_vec_2); // Zmm2 = in_val * vec_2
        Zmm3 = _mm512_sub_ps(Zmm1, Zmm2); // Zmm3 = Zmm1 - Zmm2
        _mm512_store_ps(pOutVector + i + 16, Zmm3);
    }
    std::clock_t end = std::clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time: " << elapsed << " seconds\n";

    for (int i = 0; i < 32; i++) {
        std::cout << pOutVector[i] << " ";
        if ((i + 1) % 16 == 0) {
            std::cout << std::endl;
        }
    }
    _mm_free(pInVector);
    _mm_free(pOutVector);
    return 0;
}