#pragma once
#include <vector>
#include <thread>
#include <Eigen/Dense>

using namespace Eigen;

class MatrixCal_By_ARow {
public:
    MatrixCal_By_ARow(const MatrixXf& A, const MatrixXf& B) {
        _A = A;
        _B = B;
        _Acols = A.cols();
    }
    void operator()(int start_row, int rows_per_thread, MatrixXf* block) {
        *block = _A.block(start_row, 0, rows_per_thread, _Acols) * _B;
    }
private:
    MatrixXf _A;
    MatrixXf _B;
    int _Acols;
};

class MatrixCal_By_BCol {
public:
    MatrixCal_By_BCol(const MatrixXf& A, const MatrixXf& B) {
        _A = A;
        _B = B;
        _Brows = B.rows();
    }
    void operator()(int start_col, int cols_per_thread, MatrixXf* block) {
        *block = _A * _B.block(0, start_col, _Brows, cols_per_thread);
    }
private:
    MatrixXf _A;
    MatrixXf _B;
    int _Brows;
};
