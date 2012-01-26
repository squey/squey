#ifndef GLTEST_TOOLS_H
#define GLTEST_TOOLS_H

#include <vector>
#include <pvkernel/core/PVMatrix.h>

#include <QString>

// Benchmarking macro helpers
#define START_BENCH(var) \
	tbb::tick_count __bench_##var##_start = tbb::tick_count::now();
#define END_BENCH(var, msg) \
	tbb::tick_count __bench_##var##_end = tbb::tick_count::now();\
	PVLOG_INFO(msg " %0.6f seconds", (end-start).seconds());

// Matrix typedefs
typedef PVMatrix<float, PVRow, PVCol> matrix_plotted_t;
typedef PVMatrix<float, PVRow, PVCol>::transposed_type matrix_transp_plotted_t;

bool get_plotted_as_vector(QString const& file, std::vector<float>& plotted, bool get_as_transp);
bool get_plotted_as_matrix(QString const& file, matrix_plotted_t& plotted);
bool get_plotted_as_matrix(QString const& file, matrix_trans_plotted_t& plotted);

#endif
