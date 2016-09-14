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
	PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);

	// Pre-discovery
	PVRush::list_creators pre_discovered_c =
	    PVRush::PVSourceCreatorFactory::filter_creators_pre_discovery(lcr, file);

	PVRush::PVSourceCreator_p sc_file;
	if (pre_discovered_c.size() == 0) {
		std::cerr << "No source plugins can open the file " << qPrintable(file->human_name())
		          << std::endl;
		return false;
	}
	if (pre_discovered_c.size() == 1) {
		sc_file = *(pre_discovered_c.begin());
	} else {
		// Take the source creator that have the highest success rate with the given format
		float success_rate = -1;
		PVRush::list_creators::const_iterator itc;
		for (itc = pre_discovered_c.begin(); itc != pre_discovered_c.end(); itc++) {
			PVRush::PVSourceCreator_p sc = *itc;
			PVRush::pair_format_creator fcr(format, sc);
			float sr_tmp;
			try {
				sr_tmp = PVRush::PVSourceCreatorFactory::discover_input(fcr, file);
			} catch (...) {
				continue;
			}
			if (sr_tmp > success_rate) {
				success_rate = sr_tmp;
				sc_file = sc;
			}
		}
		if (!sc_file) {
			// Take the first one
			sc_file = pre_discovered_c.front();
		}
		std::cerr << "Chose source creator '" << qPrintable(sc_file->name())
		          << "' with success rate of " << success_rate << std::endl;
	}

	sc = sc_file;

	return true;
}
