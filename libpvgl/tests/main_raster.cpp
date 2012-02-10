#include <common/common.h>
#include <common/bench.h>

#include <ocl/raster.h>

#include <picviz/PVPlotted.h>

#include <iostream>

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
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " n";
		return 1;
	}

	PVRow n = atoll(argv[1]);

	std::cout << "Creating random plotted w/ " << n << " lines..." << std::endl;
	Picviz::PVPlotted::plotted_table_t trans_plotted;
	PVCol ncols;
	if (!Picviz::PVPlotted::load_buffer_from_file(trans_plotted, ncols, true, QString("plotted.test_petit"))) {
		std::cerr << "Unable to load plotted !" << std::endl;
		return 1;
	}
	n = trans_plotted.size()/ncols;
	//init_rand_plotted(trans_plotted, n);
	
	std::cout << "Random plotted created." << std::endl;

	unsigned int* img_idxes;
	posix_memalign((void**) &img_idxes, 16, SIZE_GLOBAL_IDX_TABLE);
	ocl_raster("ocl/raster.cl", &trans_plotted.at(0), &trans_plotted.at(n), n, img_idxes, 2048.0f, 2048.0f);

	free(img_idxes);

	return 0;
}
