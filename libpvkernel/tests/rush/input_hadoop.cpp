/**
 * \file input_hadoop.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVFormat.h>

#include "../../plugins/common/hdfs/PVInputHDFSFile.h"

#include "test-env.h"
#include "helpers.h"

#include <QString>
#include <QCoreApplication>
#include <iostream>

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " format" << std::endl;
		return 1;
	}

	QCoreApplication app(argc, argv);
	init_env();
	PVRush::PVPluginsLoad::load_all_plugins();
	PVFilter::PVPluginsLoad::load_all_plugins();

	PVRush::PVFormat format("format", argv[1]);
	if (!format.populate(true)) {
		std::cerr << "Unable to populate format." << std::endl;
		return 1;
	}

	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("hdfs");
	PVRush::PVSourceCreator_p cr_text = LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("text_hdfs");

	assert(in_t);
	assert(cr_text);

	PVRush::hash_formats formats, new_formats;
	QString choseFormat;
	PVRush::PVInputType::list_inputs inputs;
	assert(in_t->createWidget(formats, new_formats, inputs, choseFormat, NULL));

	PVRush::PVInputHDFSFile* ihdfs = dynamic_cast<PVRush::PVInputHDFSFile*>(inputs[0].get());
	assert(ihdfs);
	ihdfs->set_process_in_hadoop(true);

	PVRush::PVRawSourceBase::p_type src = cr_text->create_source_from_input(inputs[0], format);
	assert(src);

	// Read chunks
	PVCore::PVChunk* read;
	while ((read = (*src)()) != NULL) {
		// And print them
		dump_chunk_csv(*read);
		read->free();
	}

	return 0;
}
