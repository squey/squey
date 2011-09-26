#define SIMULATE_PIPELINE
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/rush/PVNraw.h>


#include <cstdlib>
#include <iostream>

#include <QCoreApplication>

using std::cout;
using std::cerr;
using std::endl;

void dump_nraw_csv(PVRush::PVNraw& nraw_)
{
	PVRush::PVNraw::nraw_table &nraw = nraw_.get_table();
	PVRush::PVNraw::nraw_table::iterator it_nraw;
	PVRush::PVNraw::nraw_table_line::iterator it_nraw_line, it_nraw_line_end;
	for (it_nraw = nraw.begin(); it_nraw != nraw.end(); it_nraw++) {
		PVRush::PVNraw::nraw_table_line &l = *it_nraw;
		if (l.size() == 1) {
			QString &l_str = *(l.begin());
			std::cout << l_str.toUtf8().constData() << std::endl;
			continue;
		}
		it_nraw_line_end = l.end();
		it_nraw_line_end--;
		// for (it_nraw_line = l.begin(); it_nraw_line != it_nraw_line_end; it_nraw_line++) {
		// 	QString &field = *it_nraw_line;
		// 	std::cout << "'FOO" << field.toUtf8().constData() << "',BAR";
		// }
		// QString &field = *it_nraw_line;
		// std::cout << "'CAM" << field.toUtf8().constData() << "'NADA" << std::endl;
	}
}


int main(int argc, char** argv)
{
	if (argc <= 2) {
		cerr << "Usage: " << argv[0] << " file format" << endl;
		return 1;
	}

	QCoreApplication app(argc, argv);
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();

	// Input file
	QString path_file(argv[1]);
	PVCore::PVArgument file(path_file);

	// Load the given format file
	QString path_format(argv[2]);
	PVRush::PVFormat format("format", path_format);
	if (!format.populate(true)) {
		std::cerr << "Can't read format file " << qPrintable(path_format) << std::endl;
		return 1;
	}

	// Get the source creator
	QString file_path(argv[1]);
	PVRush::PVSourceCreator_p sc_file;
	if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
		return 1;
	}

	// Process that file with the found source creator thanks to the extractor
	PVRush::PVSourceCreator::source_p src = sc_file->create_source_from_input(file, format);
	if (!src) {
		std::cerr << "Unable to create PVRush source from file " << argv[1] << std::endl;
		return 1;
	}

	// Create the extractor
	PVRush::PVExtractor ext;
	ext.start_controller();
	ext.add_source(src);
	ext.set_chunk_filter(format.create_tbb_filters());

	// Ask for 1 million lines
	PVRush::PVControllerJob_p job = ext.process_from_agg_nlines(0, 1000000);
	job->wait_end();

	// Dump the NRAW to stdout
	// dump_nraw_csv(ext.get_nraw());

	return 0;
}
