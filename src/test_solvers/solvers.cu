#include <cuda_runtime.h>
#include <cusp/krylov/gmres.h>
#include <cusp/monitor.h>
#include <cusp/print.h>

#include <iostream>

#include "solvers.h"
// where to perform the computation
typedef cusp::device_memory MemorySpace;

std::vector<float> cudaSolver(const std::vector<int>* rowIndex,
                              const std::vector<int>* columnIndex,
                              const std::vector<float>* value,
                              const std::vector<float>* b)
{
  std::vector<float> x;
  x.resize(b->size());

  // Set handler
  cusolverSpHandle_t handleCusolver = NULL;
  cusparseHandle_t handleCusparse = NULL;
  cudaStream_t stream = NULL;

  // Initialize handler
  CUSOLVER_ERROR(cusolverSpCreate(&handleCusolver));
  CUSPARSE_ERROR(cusparseCreate(&handleCusparse));
  CUDA_ERROR(cudaStreamCreate(&stream));
  CUSOLVER_ERROR(cusolverSpSetStream(handleCusolver, stream));
  CUSPARSE_ERROR(cusparseSetStream(handleCusparse, stream));

  int *d_rowIndex, *d_columnIndex;
  float *d_value, *d_b, *d_x;  // d_p is some mediate vector
  int nnz = rowIndex->size();  // Number of non-zero values in A
  int m = b->size();           // Rows of the matrx A
  float tol = 1.e-10;
  int reorder = 0;
  int singularity = 0;

  // Allocate space on device
  CUDA_ERROR(cudaMalloc((void**) &d_rowIndex, sizeof(int) * nnz));
  CUDA_ERROR(cudaMalloc((void**) &d_columnIndex, sizeof(int) * nnz));
  CUDA_ERROR(cudaMalloc((void**) &d_value, sizeof(float) * nnz));
  CUDA_ERROR(cudaMalloc((void**) &d_b, sizeof(float) * m));
  CUDA_ERROR(cudaMalloc((void**) &d_x, sizeof(float) * m));

  // Copy data (COO storage method)
  CUDA_ERROR(cudaMemcpy(
      d_rowIndex, rowIndex->data(), sizeof(int) * nnz, cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(d_columnIndex,
                        columnIndex->data(),
                        sizeof(int) * nnz,
                        cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(
      d_value, value->data(), sizeof(float) * nnz, cudaMemcpyHostToDevice));
  CUDA_ERROR(
      cudaMemcpy(d_b, b->data(), sizeof(float) * m, cudaMemcpyHostToDevice));
  CUDA_ERROR(
      cudaMemcpy(d_x, x.data(), sizeof(float) * m, cudaMemcpyHostToDevice));

  // Create and define cusparse descriptor
  cusparseMatDescr_t descrA = NULL;
  CUSPARSE_ERROR(cusparseCreateMatDescr(&descrA));
  CUSPARSE_ERROR(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_ERROR(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

  // COO to CSR
  int* d_csrRowIndex = NULL;
  CUDA_ERROR(cudaMalloc((void**) &d_csrRowIndex,
                        sizeof(int) * (m + 1)));  // Array length: (m+1)
  CUSPARSE_ERROR(cusparseXcoo2csr(handleCusparse,
                                  d_rowIndex,
                                  nnz,
                                  m,
                                  d_csrRowIndex,
                                  CUSPARSE_INDEX_BASE_ZERO));

  // QR method
  // https://docs.nvidia.com/cuda/cusolver/index.html
  CUSOLVER_ERROR(cusolverSpScsrlsvqr(handleCusolver,
                                     m,
                                     nnz,
                                     descrA,
                                     d_value,
                                     d_csrRowIndex,
                                     d_columnIndex,
                                     d_b,
                                     tol,
                                     reorder,
                                     d_x,
                                     &singularity));

  // Copy data
  CUDA_ERROR(cudaMemcpyAsync(
      x.data(), d_x, sizeof(float) * m, cudaMemcpyDeviceToHost, stream));

  // Release space on the GPU
  CUDA_ERROR(cudaFree(d_rowIndex));
  CUDA_ERROR(cudaFree(d_columnIndex));
  CUDA_ERROR(cudaFree(d_value));
  CUDA_ERROR(cudaFree(d_b));
  CUDA_ERROR(cudaFree(d_x));
  CUDA_ERROR(cudaFree(d_csrRowIndex));
  return x;
}

std::vector<float> cuspSolver(const std::vector<int>* rowIndex,
                              const std::vector<int>* columnIndex,
                              const std::vector<float>* value,
                              const std::vector<float>* b)
{
  std::vector<float> x = {1, 1};

  int *d_rowIndex, *d_columnIndex;
  float *d_value, *d_b, *d_x;  // d_p is some mediate vector
  int nnz = rowIndex->size();  // Number of non-zero values in A
  int m = b->size();           // Rows of the matrx A

  // Allocate space on device
  CUDA_ERROR(cudaMalloc((void**) &d_rowIndex, sizeof(int) * nnz));
  CUDA_ERROR(cudaMalloc((void**) &d_columnIndex, sizeof(int) * nnz));
  CUDA_ERROR(cudaMalloc((void**) &d_value, sizeof(float) * nnz));
  CUDA_ERROR(cudaMalloc((void**) &d_b, sizeof(float) * m));
  CUDA_ERROR(cudaMalloc((void**) &d_x, sizeof(float) * m));

  // Copy data (COO storage method)
  CUDA_ERROR(cudaMemcpy(
      d_rowIndex, rowIndex->data(), sizeof(int) * nnz, cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(d_columnIndex,
                        columnIndex->data(),
                        sizeof(int) * nnz,
                        cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(
      d_value, value->data(), sizeof(float) * nnz, cudaMemcpyHostToDevice));
  CUDA_ERROR(
      cudaMemcpy(d_b, b->data(), sizeof(float) * m, cudaMemcpyHostToDevice));
  CUDA_ERROR(
      cudaMemcpy(d_x, x.data(), sizeof(float) * m, cudaMemcpyHostToDevice));
  thrust::device_ptr<int> p_rowInd(d_rowIndex);
  thrust::device_ptr<int> p_colInd(d_columnIndex);
  thrust::device_ptr<float> p_val(d_value);
  thrust::device_ptr<float> p_x(d_x);
  thrust::device_ptr<float> p_b(d_b);

  // use array1d_view to wrap the individual arrays
  typedef typename cusp::array1d_view<thrust::device_ptr<int>>
      DeviceIndexArrayView;
  typedef typename cusp::array1d_view<thrust::device_ptr<float>>
      DeviceValueArrayView;
  DeviceIndexArrayView row_indices(p_rowInd, p_rowInd + nnz);
  DeviceIndexArrayView column_indices(p_colInd, p_colInd + nnz);
  DeviceValueArrayView values(p_val, p_val + nnz);
  DeviceValueArrayView b_x(p_x, p_x + m);
  DeviceValueArrayView b_b(p_b, p_b + m);

  // combine the three array1d_views into a coo_matrix_view
  typedef cusp::coo_matrix_view<DeviceIndexArrayView,
                                DeviceIndexArrayView,
                                DeviceValueArrayView>
      DeviceView;

  // construct a coo_matrix_view from the array1d_views
  DeviceView b_A(m, m, nnz, row_indices, column_indices, values);

  // set stopping criteria:
  //  iteration_limit    = 100
  //  relative_tolerance = 1e-5
  //  absolute_tolerance = 0
  //  verbose            = true
  cusp::monitor<float> monitor(b_b, 100, 1e-10, 0, true);

  // solve the linear system A * x = b with the Conjugate Gradient method
  cusp::krylov::gmres(b_A, b_x, b_b, 50, monitor);
  cusp::print(b_A);
  cusp::print(b_x);
  cusp::print(b_b);

  // Copy data
  CUDA_ERROR(cudaMemcpyAsync(
      x.data(), d_x, sizeof(float) * m, cudaMemcpyDeviceToHost));

  // Release space on the GPU
  CUDA_ERROR(cudaFree(d_rowIndex));
  CUDA_ERROR(cudaFree(d_columnIndex));
  CUDA_ERROR(cudaFree(d_value));
  CUDA_ERROR(cudaFree(d_b));
  CUDA_ERROR(cudaFree(d_x));
  return x;
}
