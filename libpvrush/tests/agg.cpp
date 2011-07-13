#define SIMULATE_PIPELINE
#include <pvrush/PVInputFile.h>
#include <pvrush/PVChunkAlignUTF16Char.h>
#include <pvfilter/PVChunkFilter.h>
#include <pvrush/PVChunkTransformUTF16.h>
#include <pvrush/PVUnicodeSource.h>
#include <pvrush/PVAggregator.h>
#include <cstdlib>
#include <iostream>
#include <QString>
#include <QCoreApplication>
#include <QDir>
#include "helpers.h"

using std::cout;
using std::cerr;
using std::endl;

using namespace PVRush;
using namespace PVCore;

void dump_agg(PVAggregator& agg)
{
	PVChunk* pc = agg();
	while (pc) {
		dump_chunk_csv(*pc);
		pc->free();
		pc = agg();
	}
}

void show_src_index(PVAggregator& agg, size_t index)
{
	size_t offset = 0;
	PVFilter::PVRawSourceBase_p src = agg.agg_index_to_source(index, &offset);
	QFileInfo fi(src->human_name());
	// Output in UTF8 !
	cout << "Index " << index << " for source " << fi.fileName().toUtf8().constData() << " at offset " << offset << endl;
}

int main(int argc, char** argv)
{
	if (argc <= 2) {
		cerr << "Usage: " << argv[0] << " chunk_size directory" << endl;
		cerr << "Uses files in 'directory'" << endl;
		return 1;
	}
	QCoreApplication(argc, argv);

	PVFilter::PVChunkFilter null;

	QString dir_path = argv[2];
	QDir dir_files(dir_path);
	dir_files.setFilter(QDir::Files | QDir::Readable);
	QStringList files = dir_files.entryList(QStringList() << QString("*"));
	const int chunk_size = atoi(argv[1]);
	PVAggregator agg;
	for (int i = 0; i < files.size(); i++) {
		PVInput_p in(new PVInputFile(qPrintable(dir_files.absoluteFilePath(files[i]))));
		PVFilter::PVRawSourceBase_p source(new PVUnicodeSource<>(in, chunk_size, null));
		agg.add_input(source);
	}
	
	agg.read_all_chunks_from_beggining();
	agg.debug();

	show_src_index(agg, 0);
	show_src_index(agg, 1);
	show_src_index(agg, 100);
	show_src_index(agg, 250);
	show_src_index(agg, 420);
	show_src_index(agg, 550);
	
	cout << "Process from 0 to 100..." << endl;
	agg.process_indexes(0, 100);
	dump_agg(agg);

	cout << "Process from 100 to 500..." << endl;
	agg.process_indexes(100,500);
	dump_agg(agg);

	cout << "Show 1000000 lines..." << endl;
	agg.process_indexes(0,1000000);
	dump_agg(agg);

	return 0;
}
