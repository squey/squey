#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include "../../plugins/common/hdfs/PVInputHDFSFile.h"

#include "test-env.h"

#include <QString>
#include <iostream>

int main()
{
	init_env();
	PVRush::PVPluginsLoad::load_all_plugins();

	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("hdfs");
	PVRush::PVSourceCreator_p cr_text = LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("text_hdfs");

	assert(in_t);
	assert(cr_text);

	PVRush::hash_formats formats;
	QString choseFormat;
	PVRush::PVInputType::list_inputs inputs;
	assert(in_t->createWidget(formats, inputs, choseFormat, NULL));

	PVRush::PVInputHDFSFile ihdfs = inputs[0].value<PVRush::PVInputHDFSFile>();
	ihdfs.set_process_in_hadoop(true);
	PVCore::PVArgument arg;
	arg.setValue<PVRush::PVInputHDFSFile>(ihdfs);

	PVRush::PVFormat format;
	PVRush::PVRawSourceBase::p_type src = cr_text->create_source_from_input(arg, format);
	assert(src);

	// Read chunks
	PVCore::PVChunk* read;
	while ((read = (*src)()) != NULL) {
		// And print them
		fwrite(read->begin(), read->size(), 1, stdout);
		printf("\n");

		read->free();
	}

	return 0;
}
