#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <iostream>


bool PVRush::PVTests::get_file_sc(PVCore::PVArgument const& file, PVRush::PVFormat const& format, PVSourceCreator_p &sc)
{
	// Load source plugins that take a file as input
	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file");
	if (!in_t) {
		std::cerr << "Unable to load the file input type plugin !" << std::endl;
		return false;
	}
	PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);

	// Pre-discovery
	PVRush::list_creators::iterator itc;
	PVRush::list_creators pre_discovered_c;
	for (itc = lcr.begin(); itc != lcr.end(); itc++) {
		PVRush::PVSourceCreator_p sc = *itc;
		if (sc->pre_discovery(file)) {
			pre_discovered_c.push_back(sc);
		}
	}

	PVRush::PVSourceCreator_p sc_file;
	if (pre_discovered_c.size() == 0) {
		std::cerr << "No source plugins can open the file " << qPrintable(file.toString()) << std::endl;
		return false;
	}
	if (pre_discovered_c.size() == 1) {
		sc_file = *(pre_discovered_c.begin());
	}
	else {
		// Take the source creator that have the highest success rate with the given format
		float success_rate = -1;
		for (itc = pre_discovered_c.begin(); itc != pre_discovered_c.end(); itc++) {
			PVRush::PVSourceCreator_p sc = *itc;
			PVRush::pair_format_creator fcr(format, sc);
			float sr_tmp = PVRush::PVSourceCreatorFactory::discover_input(fcr, file);
			if (sr_tmp > success_rate) {
				success_rate = sr_tmp;
				sc_file = sc;
			}
		}
		std::cerr << "Chose source creator '" << qPrintable(sc_file->name()) << "' with success rate of " << success_rate << std::endl;
	}

	sc = sc_file;

	return true;
}
