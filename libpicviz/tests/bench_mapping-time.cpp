#include <picviz/PVMappingFilter.h>
#include <picviz/PVRoot.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <tbb/tick_count.h>
#include "test-env.h"

#include <QVector>
#include <QString>

#include <iostream>

#define MIN_SIZE 10
int main(int argc, char** argv)
{
	long size = (argc > 1) ? atol(argv[1]) : MIN_SIZE;
	if (size < MIN_SIZE) {
		size = MIN_SIZE;
	}
	init_env();

	Picviz::PVRoot root; // Load plugins
	Picviz::PVMappingFilter::p_type map_filter = LIB_CLASS(Picviz::PVMappingFilter)::get().get_class_by_name("time_24h");
	if (!map_filter) {
		std::cerr << "Unable to load the time_24h plugin !" << std::endl;
		return 1;
	}

	// Generate a list of "size" dates and the corresponding results and show the time taken
	PVRush::PVNraw::nraw_table_line times;
	times.resize(size);
	QString d = QString("02/Jan/2007:03:56:21");
	for (int i = 0; i < size-1; i++) {
		times[i] = d;
	}
	times[size-1] = "57381";

	QStringList time_formats;
	time_formats << QString("d/MMM/yyyy:h:m:s") << QString("epoch");
	PVRush::PVFormat format;
	format.time_format[1] = time_formats;
	map_filter->set_format(0, format);

	float* res = (float*) malloc(size * sizeof(float));
	map_filter->set_dest_array(size, res);

	tbb::tick_count start = tbb::tick_count::now();
	map_filter->operator()(times);
	tbb::tick_count end = tbb::tick_count::now();

	for (long i = 0; i < MIN_SIZE; i++) {
		std::cout << "Float value " << i << ": " << res[i] << std::endl;
	}
	free(res);

	std::cerr << "Time mapping took " << (end-start).seconds() << " seconds.\n" << std::endl;

	return 0;
}
