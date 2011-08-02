#include "test-env.h"
#include <pvkernel/filter/PVFilterLibrary.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/filter/PVFieldsFilter.h>

#include <iostream>
#include <QString>

int main()
{
	init_env();

	PVFilter::PVPluginsLoad::load_all_plugins();

	LIB_CLASS(PVFilter::PVFieldsFilter<PVFilter::one_to_many>)::list_filters const& lf = LIB_CLASS(PVFilter::PVFieldsFilter<PVFilter::one_to_many>)::get().get_list();
	LIB_CLASS(PVFilter::PVFieldsFilter<PVFilter::one_to_many>)::list_filters::const_iterator it;
	PVFilter::PVFieldsSplitter_p regexp = LIB_CLASS(PVFilter::PVFieldsFilter<PVFilter::one_to_many>)::get().get_filter_by_name("regexp");
	assert(regexp);
	{
		std::cout << "Splitters:" << std::endl;
		for (it = lf.begin(); it != lf.end(); it++) {
			std::cout << qPrintable(it.key()) << std::endl;
		}
	}

	{
		PVFilter::PVFilterLibrary<PVFilter::PVFieldsFilterReg>::get().register_filter<PVFilter::PVFieldsSplitter>("test", *regexp);
		PVFilter::PVFilterLibrary<PVFilter::PVFieldsFilterReg>::list_filters const& lf = PVFilter::PVFilterLibrary<PVFilter::PVFieldsFilterReg>::get().get_list();
		PVFilter::PVFilterLibrary<PVFilter::PVFieldsFilterReg>::list_filters::const_iterator it;

		std::cout << "All fields filters:" << std::endl;
		for (it = lf.begin(); it != lf.end(); it++) {
			std::cout << qPrintable(it.key()) << std::endl;
		}
	}


	return 0;
}
