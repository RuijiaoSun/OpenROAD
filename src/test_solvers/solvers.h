#include <cuda_runtime.h>
#include <math.h>

#include <iostream>
#include <vector>

#include "cusolverDn.h"
#include "cusolverSp.h"
#include "cusparse.h"

#define CUDA_ERROR(value)                         \
  do {                                            \
    cudaError_t m_cudaStat = value;               \
    if (m_cudaStat != cudaSuccess) {              \
      fprintf(stderr,                             \
              "Error %s at line %d in file %s\n", \
              cudaGetErrorString(m_cudaStat),     \
              __LINE__,                           \
              __FILE__);                          \
      exit(-1);                                   \
    }                                             \
  } while (0)

#define CUSPARSE_ERROR(value)                     \
  do {                                            \
    cusparseStatus_t _m_status = value;           \
    if (_m_status != CUSPARSE_STATUS_SUCCESS) {   \
      fprintf(stderr,                             \
              "Error %d at line %d in file %s\n", \
              (int) _m_status,                    \
              __LINE__,                           \
              __FILE__);                          \
      exit(-5);                                   \
    }                                             \
  } while (0)

#define CUBLAS_ERROR(value)                                                 \
  do {                                                                      \
    cublasStatus_t status = value;                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "Error at line %d in file %s\n", __LINE__, __FILE__); \
      exit(-5);                                                             \
    }                                                                       \
  } while (0)

#define CUSOLVER_ERROR(value)                     \
  do {                                            \
    cusolverStatus_t _m_status = value;           \
    if (_m_status != CUSOLVER_STATUS_SUCCESS) {   \
      fprintf(stderr,                             \
              "Error %d at line %d in file %s\n", \
              (int) _m_status,                    \
              __LINE__,                           \
              __FILE__);                          \
      exit(-5);                                   \
    }                                             \
  } while (0)

std::vector<float> cudaSolver(const std::vector<int>* rowIndex,
                              const std::vector<int>* columnIndex,
                              const std::vector<float>* value,
                              const std::vector<float>* b);
std::vector<float> cuspSolver(const std::vector<int>* rowIndex,
                              const std::vector<int>* columnIndex,
                              const std::vector<float>* value,
                              const std::vector<float>* b);
