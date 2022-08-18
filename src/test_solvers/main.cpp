#include "solvers.h"
int main()
{
  // Initialize triplets for the matrix
  std::vector<int> rowIndex = {0, 1, 1};
  std::vector<int> columnIndex = {1, 0, 1};
  std::vector<float> value = {1.0, 2.0, 3.0};
  // b is the right-hand-side vector
  std::vector<float> b = {1.7, 8.3};

  // test the iterative CUSP solver
  std::vector<float> x = cuspSolver(&rowIndex, &columnIndex, &value, &b);

  // // test the direct CUDA solver
  // std::vector<float> x = cudaSolver(&rowIndex, &columnIndex, &value, &b);
  std::cout << x[0] << "," << x[1] << std::endl;

  return 0;
}