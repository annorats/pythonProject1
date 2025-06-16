#include <demolibrarysource.h>

int doubleit(double input, double *output) {
    *output = input * 2;
    return 0;
}

int doubleitarray(int len1, double* vec1, int len2, double* vec2) {
    int acount;
    for (acount = 0; acount < len1; acount++) {
        vec2[acount] = vec1[acount] * 2;
    }
    return 0;
} 
