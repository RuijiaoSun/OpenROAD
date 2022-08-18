#include "solvers.h"
int main(){

    // Initialize triplets for the matrix
    std::vector<int> rowInd;
    std::vector<int> colInd;
    std::vector<float> val;
    rowInd.push_back(0);
    rowInd.push_back(1);
    rowInd.push_back(1);
    colInd.push_back(1);
    colInd.push_back(0);
    colInd.push_back(1);
    val.push_back(1);
    val.push_back(2);
    val.push_back(3);

    std::vector<float> b;
    std::vector<float> x;
    
    b.push_back(1.7);
    b.push_back(8.3);

    x.resize(b.size());
    // test the iterative CUSP solver
    x = cuspSolver(&rowInd, &colInd, &val, &b);
    // // test the direct CUDA solver
    // x = cudaSolver(&rowInd, &colInd, &val, &b);
    std::cout << x[0] << "," << x[1] << std::endl;

    return 0;
}