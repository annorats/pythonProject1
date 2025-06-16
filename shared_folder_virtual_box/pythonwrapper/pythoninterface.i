%module demomathslibrary 

%{
    #define SWIG_FILE_WITH_INIT
    #include "./demolibrarysource.h"     
%}

%include "./numpy.i"

%init %{
    import_array();
%}

%include "./demolibrarysource.h" 

%rename (doubleit) pythondoubleit;

%inline %{
    double pythondoubleit(double x) {
        double result;
        doubleit(x,&result);
        return result;
    }
%}

%apply (int DIM1, double* INPLACE_ARRAY1) {(int len1, double* vec1), (int len2, double* vec2)} 

%rename (doublearray) pythondoublearray;

%inline %{
    int pythondoublearray(int len1, double* vec1, int len2, double* vec2) {
        return doubleitarray(len1, vec1, len2, vec2);
    }
%}
