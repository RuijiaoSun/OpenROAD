## Purpose
This is a test project aiming to test the direct CUDA solver (https://docs.nvidia.com/cuda/cusolver/index.html) and the iterative CUSP solver (https://github.com/cusplibrary/cusplibrary) on a 2 * 2 matrix before we apply them on OpenROAD project. 

## Process
The direct CUDA solver extracts the COO format, {
  rowIndex, colIndex, value}, transforms the COO format to the CSR, and solves it using cuSolver. The cusolver function uses QR decomposition to get the inversion of the matrix, and multiply it with the RHS vector.
The iterative CUSP solver refines the initial solution in each iteration to approach the real answer.

## Build
```
mkdir build
cd build
cmake ..
make 
```
## Run
```
./small
```

## Example
The default solver is the iterative CUSP solver. The example is A \* x = b:
```
    A = [ 0  1 ]    x = [ ? ]     b = [ 1.7 ]
        [ 2  3 ]        [ ? ]         [ 8.3 ]
```
The correct answer is:
```
    x = [ 1.6 ] 
        [ 1.7 ]   
```
You can switch it to the direct CUDA solver easily in the main() function.

## Result
The time for the direct CUDA solver is 1.224s, while the CUSP solver is 0.197s.

## Conclusion
Since the scale of the matrix in OpenROAD might be more than 10k and the direct CUDA solver is good at solving big matrices, we should test it on the OpenROAD project. Also, we see the acceleration of the CUSP solver corresponding to the direct CUDA solver. We should also try the iterative CUSP solver.
