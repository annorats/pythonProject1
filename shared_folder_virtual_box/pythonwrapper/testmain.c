#include <stdio.h>
#include <demolibrarysource.h>

int main(void) {
    double myinput = 3.0;
    double result;
    double inputdata[10]={1,2,3,4,5,6,7,8,9,10};
    double outputdata[10];
    int dcount;
    doubleit(myinput, &result);
    printf("Doubling an input data element in C.\n");
    printf("Twice 3.0 is %f\n", result);
    printf("Doubling an input data array in C.\n");
    doubleitarray(10, inputdata, 10, outputdata);
    for(dcount=0; dcount<10; dcount++) {
        printf("Twice %f is %f\n", inputdata[dcount], outputdata[dcount]);
    }
    return 0;    
}