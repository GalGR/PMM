#pragma once

#include "GMM_Macros.h"
#include "Limit_Util.h"



namespace MatlabGMMDataExchange
{


	int SetEngineSparseMatrix(const char* name, GMMSparseRowMatrix& A);
	int SetEngineSparseMatrix(const char* name, GMMSparseComplexRowMatrix& A);
	int SetEngineDenseMatrix(const char* name, GMMDenseColMatrix& A);
	int SetEngineDenseMatrix(const char* name, GMMDenseComplexColMatrix& A);


	int GetEngineDenseMatrix(const char* name, GMMDenseComplexColMatrix& A);
	int GetEngineDenseMatrix(const char* name, GMMDenseColMatrix& A);
	int GetEngineSparseMatrix(const char* name, GMMSparseRowMatrix& A);
	int GetEngineSparseMatrix(const char* name, GMMSparseComplexRowMatrix& A);
	int GetEngineCompressedSparseMatrix(const char* name, GMMCompressed0ComplexRowMatrix& A);
	int GetEngineCompressedSparseMatrix(const char* name, GMMCompressed0RowMatrix& A);

	template<class SparseRowMatrixType>
	bool isMatrixValid(const SparseRowMatrixType& M);
}


template<class SparseRowMatrixType>
bool MatlabGMMDataExchange::isMatrixValid(const SparseRowMatrixType& M)
{
	auto rowIter = mat_row_const_begin(M);
	auto rowIterEnd = mat_row_const_end(M);

	for(; rowIter != rowIterEnd; rowIter++)
	{
		auto row = gmm::linalg_traits<SparseRowMatrixType>::row(rowIter);

		auto elementIter = vect_const_begin(row);
		auto elementIterEnd = vect_const_end(row);

		for(; elementIter != elementIterEnd; elementIter++)
		{
			if(!LIMIT::isFinite(*elementIter)) //this will be true if val is NAN (not a valid number)
			{
				return false;
			}
		}
	}
	return true;
}
