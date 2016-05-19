/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <pvkernel/rush/PVAggregator.h>
#include <pvkernel/rush/PVUtils.h>

#include <pvkernel/core/inendi_assert.h>

#include <cstdlib>
#include <iostream>
#include <QString>
#include <QDir>

#include "helpers.h"
#include "common.h"

using namespace PVRush;
using namespace PVCore;

void dump_agg(PVAggregator& agg, std::ostream& out)
{
	while (PVChunk* pc = agg()) {
		dump_chunk_csv(*pc, out);
		pc->free();
	}
}

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " chunk_size directory" << std::endl;
		std::cerr << "Uses files in 'directory'" << std::endl;
		return 1;
	}
	PVCore::PVIntrinsics::init_cpuid();

	QString dir_path = argv[2];
	const std::string ref_file = std::string(argv[2]) + ".strict.out";
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

	std::string output_file = pvtest::get_tmp_filename();
	std::ofstream out(output_file);

	out << "Process from 10 to 20..." << std::endl;
	agg.process_indexes(10, 20);
	dump_agg(agg, out);

	out << "Process from 127 to 140..." << std::endl;
	agg.process_indexes(127, 140);
	dump_agg(agg, out);

	out << "Process from 0 to 4..." << std::endl;
	agg.process_indexes(0, 4);
	dump_agg(agg, out);

	out << "Process from 1 to 701..." << std::endl;
	agg.process_indexes(1, 701);
	dump_agg(agg, out);

#ifndef INSPECTOR_BENCH
	// Check output is the same as the reference
	std::cout << output_file << " - " << ref_file << std::endl;
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file, ref_file));
#endif

	return 0;
}
