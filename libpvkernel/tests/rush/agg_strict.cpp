/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <pvkernel/rush/PVAggregator.h>
#include <cstdlib>
#include <iostream>
#include <QString>
#include <QDir>
#include "helpers.h"

// FIXME: what can be asserted or measured?
using std::cout;
using std::cerr;
using std::endl;

using namespace PVRush;
using namespace PVCore;

void dump_agg(PVAggregator& agg)
{
	PVChunk* pc = agg();
	while (pc) {
		dump_chunk_csv(*pc, std::cout);
		pc->free();
		pc = agg();
	}
}

int main(int argc, char** argv)
{
	if (argc <= 2) {
		cerr << "Usage: " << argv[0] << " chunk_size directory" << endl;
		cerr << "Uses files in 'directory'" << endl;
		return 1;
	}
	PVCore::PVIntrinsics::init_cpuid();

	QString dir_path = argv[2];
	QDir dir_files(dir_path);
	dir_files.setFilter(QDir::Files | QDir::Readable);
	QStringList files = dir_files.entryList(QStringList() << QString("*"));
	const int chunk_size = atoi(argv[1]);
	PVAggregator agg;
	for (int i = 0; i < files.size(); i++) {
		PVInput_p in(new PVInputFile(qPrintable(dir_files.absoluteFilePath(files[i]))));
		PVRush::PVRawSourceBase_p source(new PVUnicodeSource<>(in, chunk_size));
		agg.add_input(source);
	}

	// Set strict mode
	agg.set_strict_mode(true);

	cout << "Process from 10 to 20..." << endl;
	agg.process_indexes(10, 20);
	dump_agg(agg);

	cout << "Process from 127 to 140..." << endl;
	agg.process_indexes(127, 140);
	dump_agg(agg);

	cout << "Process from 0 to 4..." << endl;
	agg.process_indexes(0, 4);
	dump_agg(agg);

	cout << "Process from 1 to 701..." << endl;
	agg.process_indexes(1, 701);
	dump_agg(agg);

	return 0;
}
