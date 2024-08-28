#include <iostream>
#include <ctime>

int main(int argc, char const *argv[])
{
    int len = 2100000000;

    // Dynamic memory allocation
    float *pInVector = new float[len];
    float *pOutVector = new float[len];

    // Initialize data
    for (int i = 0; i < len; i++)
        pInVector[i] = 1.0f;

    float nega = -1;
    float pos = 1;

    // Process data using a pure loop
    std::clock_t start = std::clock();
    for (int i = 0; i < len; i++) {
        float in_val = pInVector[i];

        // Compute the output value
        float out_val = (in_val * pos) - (in_val * nega);
        
        pOutVector[i] = out_val;
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
    
    delete[] pInVector;
    delete[] pOutVector;
    return 0;
}