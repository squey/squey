// B-code->reduction CPU->GPU pipeline
//

#include <common/common.h>
#include <common/bench.h>
#include <cuda/common.h>
#include <code_bz/bz_compute.h>
#include <code_bz/bcode_cb.h>
#include <code_bz/types.h>
#include <code_bz/serial_bcodecb.h>
#include <cuda/gpu_bccb.h>

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

	init_cuda();
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
	init_rand_plotted(trans_plotted, 1024*256*100);
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
	BCodeCB cb_ref = allocate_BCodeCB();
	BENCH_START(ref);
	// B-code and then serial reduction
	int ncodes = bz.compute_b_trans(&codes_ref[0], 0, 1, X_START, X_START+W_FRAME-1, Y_START, Y_START+H_FRAME-1);
	serial_bcodecb(&codes_ref[0], ncodes, cb_ref);
	BENCH_END(ref, "BCode+red computation", bz.get_nrows()*2, sizeof(float), 1, SIZE_BCODECB);
	printf("\n");


	size_t nthreads = 4;
	PVBCode* pcodes[nthreads];
	size_t nallocs = bz.get_nrows();
	nallocs = ((nallocs+3)/4)*4;
	for (size_t j = 0; j < nthreads; j++) {
		pcodes[j] = PVCore::PVAlignedAllocator<PVBCode, 16>().allocate(nallocs);
	}
	// Bootstreap openmp
	BENCH_START(omp);
	ncodes = bz.compute_b_trans_sse4_notable_omp(pcodes, 0, 1, X_START, X_START+W_FRAME-1, Y_START, Y_START+H_FRAME-1, 4);
	BENCH_END(omp, "omp", bz.get_nrows()*2, sizeof(float), 1, sizeof(BCodeCB));

	BENCH_START(omp2);
	ncodes = bz.compute_b_trans_sse4_notable_omp(pcodes, 0, 1, X_START, X_START+W_FRAME-1, Y_START, Y_START+H_FRAME-1, 4);
	BENCH_END(omp2, "omp2", bz.get_nrows()*2, sizeof(float), 1, sizeof(BCodeCB));

	BCodeCB cb_gpu = allocate_BCodeCB();
	/*
	{
		GPUBccb gpu_bccb;
		BENCH_START(pipe);
		ncodes = bz.compute_b_trans_sse4_notable_omp_pipe(0, 1, X_START, X_START+W_FRAME-1, Y_START, Y_START+H_FRAME-1, 1024*1024, gpu_bccb, 4);
		BENCH_END(pipe, "Pipe", bz.get_nrows()*2, sizeof(float), 1, sizeof(BCodeCB));
		gpu_bccb.copy_bccb_from_device(cb_gpu);
	}*/

		GPUBccb gpu_bccb;
		BENCH_START(pipe);
		ncodes = bz.compute_b_trans_sse4_notable_omp_pipe2(0, 1, X_START, X_START+W_FRAME-1, Y_START, Y_START+H_FRAME-1, 1024*8, gpu_bccb, 4);
		BENCH_END(pipe, "Pipe", bz.get_nrows()*2, sizeof(float), 1, sizeof(BCodeCB));
		gpu_bccb.copy_bccb_from_device(cb_gpu);

	/*
	{
		GPUBccb gpu_bccb;
		BENCH_START(pipe);
		ncodes = bz.compute_b_trans_sse4_notable_omp_pipe4(0, 1, X_START, X_START+W_FRAME-1, Y_START, Y_START+H_FRAME-1, 128, gpu_bccb, 4);
		BENCH_END(pipe, "Pipe", bz.get_nrows()*2, sizeof(float), 1, sizeof(BCodeCB));
		gpu_bccb.copy_bccb_from_device(cb_gpu);
	}*/


	write(4, cb_ref, SIZE_BCODECB);
	write(5, cb_gpu, SIZE_BCODECB);

	free_BCodeCB(cb_ref);
	free_BCodeCB(cb_gpu);

	return 0;
}
