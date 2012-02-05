// Benchmarking and optimisations of the B-code computation
//

#include <common/common.h>
#include <common/bench.h>
#include <code_bz/bz_compute.h>

#include <picviz/PVPlotted.h>

#include <ctime>

#define W_FRAME 2048
#define H_FRAME 2048

#define X_START 0
#define Y_START 0

#define MAX_ERR_PRINT 40

#define LAUNCH_BENCH(name, desc, f)\
	codes.clear();\
	codes.resize(bz.get_nrows());\
	BENCH_START(name);\
	bz.f(&codes[0], 0, 1, X_START, X_START+W_FRAME, Y_START, Y_START+H_FRAME);\
	BENCH_END(name, desc, bz.get_nrows()*2, sizeof(float), codes.size(), sizeof(PVBCode));\
	show_codes_diff(&codes_ref[0], &codes[0], codes.size());\
	/*CHECK(codes.size() == codes_ref.size());\
	CHECK(memcmp(&codes[0], &codes_ref[0], codes.size()*sizeof(PVBCode)) == 0);*/

void show_codes_diff(PVBCode* bref, PVBCode* bcmp, size_t n)
{
	size_t err_print = 0;
	for (size_t i = 0; i < n; i++) {
		if (bref[i].int_v != bcmp[i].int_v) {
			PVBCode ref = bref[i];
			PVBCode cmp = bcmp[i];
			if (ref.s.type != cmp.s.type || abs(ref.s.l-cmp.s.l) > 4 || abs(ref.s.r-cmp.s.r) > 4 ||
			    ref.s.__free != 0 || cmp.s.__free != 0) {
				printf("Ref: %d|%d|%d|%d ; Wrong: %d|%d|%d|%d\n", ref.s.type, ref.s.l, ref.s.r, ref.s.__free,
			                                          cmp.s.type, cmp.s.l, cmp.s.r, cmp.s.__free);
				err_print++;
				if (err_print >= MAX_ERR_PRINT) {
					printf("Too many errors. Stopping...\n");
					return;
				}
			}
		}
	}
}

void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p, PVRow nrows)
{
	srand(time(NULL));
	p.clear();
	p.reserve(nrows*2);
	for (PVRow i = 0; i < nrows; i++) {
		p.push_back((float)((double)(rand()-1)/(double)RAND_MAX));
		p.push_back((float)((double)(rand()-1)/(double)RAND_MAX));
	}
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " plotted_file" << std::endl;
		return 1;
	}

	if (sizeof(PVBCode) != sizeof(uint32_t)) {
		std::cerr << "sizeof PVBCode is different from sizeof(uint32_t) !!" << std::endl;
		return 1;
	}

	std::cout << "Reading plotted..." << std::endl;
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
	init_rand_plotted(trans_plotted, 40000000);
	std::cout << "Plotted read." << std::endl;

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
	BENCH_START(bcode_trans);
	bz.compute_b_trans(&codes_ref[0], 0, 1, X_START, X_START+W_FRAME-1, Y_START, Y_START+H_FRAME-1);
	BENCH_END(bcode_trans, "BCode trans-computation", bz.get_nrows()*2, sizeof(float), codes_ref.size(), sizeof(PVBCode));

	// Launch other benchs
	//LAUNCH_BENCH(bcode_trans_nobranch, "BCode trans-nobranch", compute_b_trans_nobranch);
	//LAUNCH_BENCH(bcode_trans_nobranch_sse, "BCode trans-nobranch-sse", compute_b_trans_nobranch_sse);
	LAUNCH_BENCH(bcode_trans_int, "BCode trans-int",  compute_b_trans_int);
	LAUNCH_BENCH(bcode_trans_sse, "BCode trans-sse",  compute_b_trans_sse);
	//LAUNCH_BENCH(bcode_trans_sse2, "BCode trans-sse2",  compute_b_trans_sse2);
	//LAUNCH_BENCH(bcode_trans_sse3, "BCode trans-sse3",  compute_b_trans_sse3);
	LAUNCH_BENCH(bcode_trans_sse4, "BCode trans-sse4",  compute_b_trans_sse4);

	return 0;
}
