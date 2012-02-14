// Benchmarking and optimisations of the B-code computation
//

#include <common/common.h>
#include <common/bench.h>
#include <code_bz/bz_compute.h>

#include <picviz/PVPlotted.h>

#include <ctime>
#include <math.h>

#include <omp.h>

#define W_FRAME 2048
#define H_FRAME 2048

#define X_START 0
#define Y_START 0

#define MAX_ERR_PRINT 40

#define LAUNCH_BENCH(name, desc, f)\
	codes.clear();\
	codes.resize(bz.get_nrows());\
	BENCH_START(name);\
	ncodes = bz.f(&codes[0], 0, 1, X_START, X_START+W_FRAME, Y_START, Y_START+H_FRAME);\
	BENCH_END(name, desc, bz.get_nrows()*2, sizeof(float), ncodes, sizeof(PVBCode));\
	{\
		double freq_types[6];\
		printf("Mean norm-2 difference: %0.4f %%.\n", stats_codes_diff(&codes_ref[0], &codes[0], codes.size(), freq_types)*100.0);\
		printf("Types frequency: ");\
		for (int i = 0; i < 6; i++) {\
			printf("%d: %0.4f %% | ", i, freq_types[i]*100.0);\
		}\
		printf("\n\n");\
	}
	/*CHECK(codes.size() == codes_ref.size());\
	CHECK(memcmp(&codes[0], &codes_ref[0], codes.size()*sizeof(PVBCode)) == 0);*/

double stats_codes_diff(PVBCode* bref, PVBCode* bcmp, size_t n, double freq_types[6])
{
	double rel_diff = 0;
	uint32_t ntypes[6];
	for (int i = 0; i < 6; i++) {
		ntypes[i] = 0.0;
	}
	for (size_t i = 0; i < n; i++) {
		if (bref[i].int_v != bcmp[i].int_v) {
			PVBCode ref = bref[i];
			PVBCode cmp = bcmp[i];
			if (ref.int_v != cmp.int_v) {
				// Compute norm-2 difference
				uint16_t ref_lx,ref_ly,ref_rx,ref_ry;
				uint16_t cmp_lx,cmp_ly,cmp_rx,cmp_ry;
				ref.to_pts(W_FRAME, H_FRAME, ref_lx, ref_ly, ref_rx, ref_ry);
				cmp.to_pts(W_FRAME, H_FRAME, cmp_lx, cmp_ly, cmp_rx, cmp_ry);
				double diff_l = (sqrt((double)((ref_ly-cmp_ly)*(ref_ly-cmp_ly) + (ref_lx-cmp_lx)*(ref_lx-cmp_lx))))/(2048.0);
				double diff_r = (sqrt((double)((ref_ry-cmp_ry)*(ref_ry-cmp_ry) + (ref_rx-cmp_rx)*(ref_rx-cmp_rx))))/(2048.0);
				rel_diff += (diff_l+diff_r)/2.0;
			}
			ntypes[cmp.s.type]++;
		}
	}

	for (int i = 0; i < 6; i++) {
		freq_types[i] = ((double)ntypes[i])/((double)n);
	}
	return rel_diff/((double)(n));
}

void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p, PVRow nrows)
{
	srand(time(NULL));
	p.clear();
	p.reserve(nrows*2);
	for (PVRow i = 0; i < nrows; i++) {
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
	}
}

int main(int argc, char** argv)
{
	
	/*if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " plotted_file" << std::endl;
		return 1;
	}*/

	if (sizeof(PVBCode) != sizeof(uint32_t)) {
		std::cerr << "sizeof PVBCode is different from sizeof(uint32_t) !!" << std::endl;
		return 1;
	}

	// OpenMP first startup takes some time, so doing useless stuff
	// here to avoid false results.
	int a = 0;
#pragma omp parallel for
	for (int i = 0; i < 100000; i++) {
		a += i;
	}

	std::cout << "Creating random plotted..." << std::endl;
	PVCol ncols = 2;
	Picviz::PVPlotted::plotted_table_t plotted, trans_plotted;
	/*if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, false, QString(argv[1]))) {
		std::cerr << "Unable to load plotted !" << std::endl;
		return 1;
	}*/
	/*if (!Picviz::PVPlotted::load_buffer_from_file(trans_plotted, ncols, true, QString(argv[1]))) {
		std::cerr << "Unable to load plotted !" << std::endl;
		return 1;
	}*/
	init_rand_plotted(trans_plotted, 400000);
	std::cout << "Random plotted created." << std::endl;

	PVBZCompute bz;
	//bz.set_plotted(plotted, ncols);
	bz.set_trans_plotted(trans_plotted, ncols);
	bz.set_zoom(2048, 2048);
	
	std::cout << "Start BCode computation..." << std::endl;
	std::vector<PVBCode, PVCore::PVAlignedAllocator<PVBCode, 16> > codes_ref, codes;

	/*BENCH_START(bcode);
	bz.compute_b(codes, 0, 1, X_START, X_START+W_FRAME, Y_START, Y_START+H_FRAME);
	BENCH_END(bcode, "BCode computation", bz.get_nrows()*2, sizeof(float), codes.size(), sizeof(PVBCode));*/

	// Compute reference
	codes_ref.resize(bz.get_nrows());
#if 0
	BENCH_START(bcode_trans);
	bz.compute_b_trans(&codes_ref[0], 0, 1, X_START, X_START+W_FRAME-1, Y_START, Y_START+H_FRAME-1);
	BENCH_END(bcode_trans, "BCode trans-computation", bz.get_nrows()*2, sizeof(float), codes_ref.size(), sizeof(PVBCode));
	printf("\n");

	// Launch other benchs
	//LAUNCH_BENCH(bcode_trans_nobranch, "BCode trans-nobranch", compute_b_trans_nobranch);
	//LAUNCH_BENCH(bcode_trans_nobranch_sse, "BCode trans-nobranch-sse", compute_b_trans_nobranch_sse);
	//LAUNCH_BENCH(bcode_trans2, "BCode trans2",  compute_b_trans2);
	LAUNCH_BENCH(bcode_trans_int, "BCode trans-int",  compute_b_trans_int);
	//LAUNCH_BENCH(bcode_trans_int_ld, "BCode trans-int-ld",  compute_b_trans_int_ld);
	LAUNCH_BENCH(bcode_trans_sse, "BCode trans-sse",  compute_b_trans_sse);
	LAUNCH_BENCH(bcode_trans_sse_int, "BCode trans-sse-int",  compute_b_trans_sse_int);
	//LAUNCH_BENCH(bcode_trans_sse2, "BCode trans-sse2",  compute_b_trans_sse2);
	//LAUNCH_BENCH(bcode_trans_sse3, "BCode trans-sse3",  compute_b_trans_sse3);
	LAUNCH_BENCH(bcode_trans_sse4, "BCode trans-sse4",  compute_b_trans_sse4);
	LAUNCH_BENCH(bcode_trans_sse4_int, "BCode trans-sse4-int",  compute_b_trans_sse4_int);
#endif

	// Reference w/ "no-table" variants
	BENCH_START(bcode_trans_notable);
	int ncodes = bz.compute_b_trans_notable(&codes_ref[0], 0, 1, X_START, X_START+W_FRAME-1, Y_START, Y_START+H_FRAME-1);
	BENCH_END(bcode_trans_notable, "BCode trans-computation notable", bz.get_nrows()*2, sizeof(float), ncodes, sizeof(PVBCode));
	LAUNCH_BENCH(bcode_trans_sse4_notable, "BCode trans-sse4-notable",  compute_b_trans_sse4_notable);

	// OMP
	//size_t nthreads = omp_get_max_threads();
	size_t nthreads = 12;
	PVBCode* pcodes[nthreads];
	size_t nallocs = (bz.get_nrows()+nthreads-1)/nthreads;
	nallocs = ((nallocs+3)/4)*4;
	for (size_t i = 0; i < nthreads; i++) {
		pcodes[i] = PVCore::PVAlignedAllocator<PVBCode, 16>().allocate(nallocs);
	}
	BENCH_START(bcode_trans_notable_omp);
	ncodes = bz.compute_b_trans_sse4_notable_omp(pcodes, 0, 1, X_START, X_START+W_FRAME-1, Y_START, Y_START+H_FRAME-1);
	BENCH_END(bcode_trans_notable_omp, "BCode trans-computation notable-omp", bz.get_nrows()*2, sizeof(float), ncodes, sizeof(PVBCode));

	BENCH_START(bcode_trans_notable_omp1);
	ncodes = bz.compute_b_trans_sse4_notable_omp(pcodes, 0, 1, X_START, X_START+W_FRAME-1, Y_START, Y_START+H_FRAME-1);
	BENCH_END(bcode_trans_notable_omp1, "BCode trans-computation notable-omp", bz.get_nrows()*2, sizeof(float), ncodes, sizeof(PVBCode));

	BENCH_START(bcode_trans_notable_omp2);
	ncodes = bz.compute_b_trans_sse4_notable_omp(pcodes, 0, 1, X_START, X_START+W_FRAME-1, Y_START, Y_START+H_FRAME-1);
	BENCH_END(bcode_trans_notable_omp2, "BCode trans-computation notable-omp", bz.get_nrows()*2, sizeof(float), ncodes, sizeof(PVBCode));

	BENCH_START(bcode_trans_notable_omp4);
	ncodes = bz.compute_b_trans_sse4_notable_omp(pcodes, 0, 1, X_START, X_START+W_FRAME-1, Y_START, Y_START+H_FRAME-1);
	BENCH_END(bcode_trans_notable_omp4, "BCode trans-computation notable-omp", bz.get_nrows()*2, sizeof(float), ncodes, sizeof(PVBCode));

	return 0;
}
