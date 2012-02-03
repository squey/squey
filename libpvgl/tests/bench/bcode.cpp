// Benchmarking and optimisations of the B-code computation
//

#include <common/common.h>
#include <common/bench.h>
#include <code_bz/bz_compute.h>

#include <picviz/PVPlotted.h>

#define W_FRAME 2048
#define H_FRAME 2048

#define X_START 1000
#define Y_START 1000

void init_rand_plotted(std::vector<float>& p, PVRow nrows)
{
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
	bz.set_zoom(8196, 8196);
	
	std::cout << "Start BCode computation..." << std::endl;
	std::vector<PVBCode> codes, trans_codes;
	codes.reserve(bz.get_nrows());
	trans_codes.reserve(bz.get_nrows());

	BENCH_START(bcode);
	//bz.compute_b(codes, 0, 1, X_START, X_START+W_FRAME, Y_START, Y_START+H_FRAME);
	BENCH_END(bcode, "BCode computation", bz.get_nrows()*2, sizeof(float), codes.size(), sizeof(PVBCode));

	BENCH_START(bcode_trans);
	bz.compute_b_trans(codes, 0, 1, X_START, X_START+W_FRAME, Y_START, Y_START+H_FRAME);
	BENCH_END(bcode_trans, "BCode trans-computation", bz.get_nrows()*2, sizeof(float), codes.size(), sizeof(PVBCode));

	//CHECK(codes.size() == trans_codes.size());
	//CHECK(memcmp(&codes[0], &trans_codes[0], codes.size()*sizeof(PVBCode)) == 0);

	trans_codes.clear();
	trans_codes.reserve(bz.get_nrows());
	BENCH_START(bcode_trans_nobranch);
	bz.compute_b_trans_nobranch(trans_codes, 0, 1, X_START, X_START+W_FRAME, Y_START, Y_START+H_FRAME);
	BENCH_END(bcode_trans_nobranch, "BCode trans-computation-nobranch", bz.get_nrows()*2, sizeof(float), trans_codes.size(), sizeof(PVBCode));

	CHECK(codes.size() == trans_codes.size());
	CHECK(memcmp(&codes[0], &trans_codes[0], codes.size()*sizeof(PVBCode)) == 0);

	trans_codes.clear();
	trans_codes.reserve(bz.get_nrows());
	BENCH_START(bcode_trans_nobranch_sse);
	bz.compute_b_trans_nobranch_sse(trans_codes, 0, 1, X_START, X_START+W_FRAME, Y_START, Y_START+H_FRAME);
	BENCH_END(bcode_trans_nobranch_sse, "BCode trans-computation-nobranch-sse", bz.get_nrows()*2, sizeof(float), trans_codes.size(), sizeof(PVBCode));

	trans_codes.clear();
	trans_codes.reserve(bz.get_nrows());
	BENCH_START(bcode_trans_sse);
	bz.compute_b_trans_sse(trans_codes, 0, 1, X_START, X_START+W_FRAME, Y_START, Y_START+H_FRAME);
	BENCH_END(bcode_trans_sse, "BCode trans-computation-sse", bz.get_nrows()*2, sizeof(float), trans_codes.size(), sizeof(PVBCode));

	CHECK(codes.size() == trans_codes.size());
	CHECK(memcmp(&codes[0], &trans_codes[0], codes.size()*sizeof(PVBCode)) == 0);

	return 0;
}
