#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVExtractor.h>
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

	PVRush::hash_formats formats, new_formats;
	QString choseFormat;
	PVRush::PVInputType::list_inputs inputs;
	PVCore::PVArgumentList tmp_args = PVCore::PVArgumentList();
	assert(in_t->createWidget(formats, new_formats, inputs, choseFormat, tmp_args, NULL));

	PVRush::PVRawSourceBase::p_type src = cr_text->create_discovery_source_from_input(inputs[0], formats.values().at(0));
	assert(src);

	// Read a chunk !
	PVCore::PVChunk* read = (*src)();

	// And print it
	PVCore::list_elts const& l = read->c_elements();
	PVCore::list_elts::const_iterator it;
	QString str_tmp;
	for (it = l.begin(); it != l.end(); it++) {
		std::cout << qPrintable((*it)->get_qstr(str_tmp)) << std::endl;
	}

	read->free();

	return 0;
}
