#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <QString>
#include <iostream>

int main()
{
	PVRush::PVPluginsLoad::load_all_plugins();

	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("hdfs");
	PVRush::PVSourceCreator_p cr_text = LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("text_hdfs");

	assert(in_t);
	assert(cr_text);

	PVRush::hash_formats formats;
	QString choseFormat;
	PVRush::PVInputType::list_inputs inputs;
	assert(in_t->createWidget(formats, inputs, choseFormat, NULL));

	PVRush::PVRawSourceBase::p_type src = cr_text->create_source_from_input(inputs[0]);
	assert(src);

	// Read a chunk !
	PVCore::PVChunk* read = (*src)();

	// And print it
	PVCore::list_elts const& l = read->c_elements();
	PVCore::list_elts::const_iterator it;
	for (it = l.begin(); it != l.end(); it++) {
		std::cout << qPrintable(it->qstr()) << std::endl;
	}

	read->free();

	return 0;
}
