/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVInputType.h>            // for PVInputType_p, PVInputType
#include <pvkernel/rush/PVSourceCreator.h>        // for PVSourceCreator_p, etc
#include <pvkernel/rush/PVSourceCreatorFactory.h> // for list_creators, etc
#include <pvkernel/rush/PVTests.h>

#include <pvkernel/core/PVClassLibrary.h> // for LIB_CLASS, etc
#include <pvkernel/core/PVRegistrableClass.h>

#include <iostream> // for operator<<, basic_ostream, etc
#include <list>     // for _List_const_iterator, etc
#include <memory>   // for __shared_ptr

namespace PVRush
{
class PVFormat;
} // namespace PVRush

bool PVRush::PVTests::get_file_sc(PVInputDescription_p file,
                                  PVRush::PVFormat const& format,
                                  PVSourceCreator_p& sc)
{
	// Load source plugins that take a file as input
	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file");
	if (!in_t) {
		std::cerr << "Unable to load the file input type plugin !" << std::endl;
		return false;
	}
	sc = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);

	return true;
}
