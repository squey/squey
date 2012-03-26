#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVMatrix.h>

#include <iostream>
#include <cstdlib>
#include <ctime>

void original_fill(float* res, const float* plotted, size_t plotted_ncols, PVRow start, PVRow end, PVCol* cols, size_t ncols)
{
	for (PVRow r = start; r < end; r++) {
		size_t offset_row_res = r*ncols;
		const float* plotted_line = &plotted[r*plotted_ncols];
		for (size_t c = 0; c < ncols; c++) {
			res[offset_row_res + c] = plotted_line[cols[c]];
		}
	}
}

void omp_fill(float* res, const float* plotted, size_t plotted_ncols, PVRow start, PVRow end, PVCol* cols, size_t ncols)
{
#pragma omp parallel for num_threads(12) 
	for (PVRow r = start; r < end; r++) {
		size_t offset_row_res = r*ncols;
		const float* plotted_line = &plotted[r*plotted_ncols];
		for (size_t c = 0; c < ncols; c++) {
			res[offset_row_res + c] = plotted_line[cols[c]];
		}
	}
}

void trans_fill(float* res, const float* plotted, size_t plotted_nrows, PVRow start, PVRow end, PVCol* cols, size_t ncols)
{
	for (PVRow r = start; r < end; r++) {
		size_t offset_row_res = r*ncols;
		for (size_t c = 0; c < ncols; c++) {
			res[offset_row_res + c] = plotted[cols[c]*plotted_nrows+r];
		}
	}
}

void trans_fill2(float* res, const float* plotted, size_t plotted_nrows, PVRow start, PVRow end, PVCol* cols, size_t ncols)
{
	for (size_t c = 0; c < ncols; c++) {
		const size_t offset_col_plotted = cols[c]*plotted_nrows;
		const float* plotted_line = &plotted[offset_col_plotted];
		for (PVRow r = start; r < end; r++) {
			size_t offset_row_res = r*ncols;
			res[offset_row_res + c] = plotted_line[r];
		}
	}
}

float* allocate_res(size_t rows, size_t cols)
{
	float* res;
	size_t s = rows*cols*sizeof(float);
	posix_memalign((void**) &res, 16, s);
	memset(res, 0, s);
	return res;
}

void init_data(PVCore::PVMatrix<float, size_t, size_t>& plotted, PVCore::PVMatrix<float, size_t, size_t>& trans_plotted, float** res, float** ref_res, size_t nrows, size_t ncols)
{
	plotted.resize(nrows, ncols);
	for (size_t i = 0; i < nrows*ncols; i++) {
		plotted.get_data()[i] = (float)rand()/(float)RAND_MAX;
	}
	plotted.transpose_to(trans_plotted);

	*ref_res = allocate_res(nrows, ncols);
	*res = allocate_res(nrows, ncols);
}

int main(int argc, char** argv)
{
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " nrows ncols" << std::endl;
		return 1;
	}

	size_t nrows = atoll(argv[1]);
	size_t ncols = atoll(argv[2]);
	if (ncols < 2) {
		ncols = 2;
	}
	std::cout << "nrows = " << nrows << ", ncols = " << ncols << std::endl;

	srand(time(NULL));

	std::vector<PVCol> all_cols;
	all_cols.reserve(ncols);
	for (size_t i = 0; i < ncols; i++) {
		all_cols.push_back(i);
	}

	float *res, *ref_res;
	{
		PVCore::PVMatrix<float, size_t, size_t> plotted,trans_plotted;
		init_data(plotted, trans_plotted, &res, &ref_res, nrows, ncols);

		BENCH_START(org);
		original_fill(ref_res, plotted.get_data(), ncols, 0, nrows, &all_cols[0], ncols);
		BENCH_END_TRANSFORM(org, "original", ncols*nrows, sizeof(float));

		BENCH_START(omp);
		omp_fill(res, plotted.get_data(), ncols, 0, nrows, &all_cols[0], ncols);
		BENCH_END_TRANSFORM(omp, "omp", ncols*nrows, sizeof(float));
		CHECK(memcmp(res, ref_res, nrows*ncols*sizeof(float)) == 0);

		BENCH_START(omp2);
		omp_fill(res, plotted.get_data(), ncols, 0, nrows, &all_cols[0], ncols);
		BENCH_END_TRANSFORM(omp2, "omp2", ncols*nrows, sizeof(float));
		CHECK(memcmp(res, ref_res, nrows*ncols*sizeof(float)) == 0);

		BENCH_START(transp);
		trans_fill(res, trans_plotted.get_data(), nrows, 0, nrows, &all_cols[0], ncols);
		BENCH_END_TRANSFORM(transp, "transp", ncols*nrows, sizeof(float));
		CHECK(memcmp(res, ref_res, nrows*ncols*sizeof(float)) == 0);

		BENCH_START(transp2);
		trans_fill2(res, trans_plotted.get_data(), nrows, 0, nrows, &all_cols[0], ncols);
		BENCH_END_TRANSFORM(transp2, "transp2", ncols*nrows, sizeof(float));
		CHECK(memcmp(res, ref_res, nrows*ncols*sizeof(float)) == 0);

		free(res); free(ref_res);
	}

	std::cout << std::endl << "Half of the columns:" << std::endl;
	{
		PVCore::PVMatrix<float, size_t, size_t> plotted,trans_plotted;
		init_data(plotted, trans_plotted, &res, &ref_res, nrows, ncols);

		BENCH_START(org);
		original_fill(ref_res, plotted.get_data(), ncols, 0, nrows, &all_cols[0], ncols/2);
		BENCH_END_TRANSFORM(org, "original", (ncols/2)*nrows, sizeof(float));

		BENCH_START(transp);
		trans_fill(res, trans_plotted.get_data(), nrows, 0, nrows, &all_cols[0], ncols/2);
		BENCH_END_TRANSFORM(transp, "transp", (ncols/2)*nrows, sizeof(float));
		CHECK(memcmp(res, ref_res, nrows*(ncols/2)*sizeof(float)) == 0);

		BENCH_START(transp2);
		trans_fill2(res, trans_plotted.get_data(), nrows, 0, nrows, &all_cols[0], ncols/2);
		BENCH_END_TRANSFORM(transp2, "transp2", (ncols/2)*nrows, sizeof(float));
		CHECK(memcmp(res, ref_res, nrows*(ncols/2)*sizeof(float)) == 0);

		free(res); free(ref_res);
	}

	return 0;
}
