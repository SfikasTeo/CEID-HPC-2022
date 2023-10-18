# Working with Openblas And Lapack on C
#### Usefull Resources and Prerequisite packages : 
* [Official OpenBlas Repository](https://github.com/xianyi/OpenBLAS)
* [Official Lapacke Docuementation](https://netlib.org/lapack/)
* [Example Repository](https://github.com/Foadsf/Cmathtuts) for Openblas and Lapacke
* Installation either manually or with packages of the libraries is mandatory :  
`pacman -S lapack lapacke lapack-doc <openblas or blas> cblas`  
## Openblas
In order to use correctly the Openblas functions with C the **cblas** interface is mandatory.  
Effectively the use of `#include <cblas.h>` and compiling with the `-lcblas` is enough to succesfully  
run blas calls.    
**In order to tune the the Openblas library use :**
```
export OPENBLAS_NUM_THREADS=threads       openblas_set_num_threads(threads);
export GOTO_NUM_THREADS=threads           goto_set_num_threads(threads);
export OMP_NUM_THREADS=threads            
```
**List of functions:**
1. cblas_zdotc_sub: dot product of two complex vectors
2. cblas_dgemm: matrix multiplication
3. cblas_dscal: scale a vector/array
4. cblas_dswap: Exchanges the elements of two vectors (double precision)
5. cblas_dcopy: Copies a vector to another vector (double-precision).
6. cblas_daxpy: Computes a vector-scalar product and adds the result to a vector.
7. cblas_dgemv: Multiplies a matrix by a vector (double precision).
8. cblas_dger: Multiplies vector X by the transform of vector Y, then adds matrix A (double precision).
9. cblas_ddot: Computes the dot product of two vectors (double-precision).  

## Lapack
Similarly with the openblas execution the inclusion of `#include <lapacke.h>` and linking with   
`-llapacke` during compilation is sufficient to run lapack calls.

## Some random Extra resources : 
* [Amd:: Developer Guides](https://developer.amd.com/resources/developer-guides-manuals/)
* [Agner:: Software Optimization Guides](https://www.agner.org/optimize/)

