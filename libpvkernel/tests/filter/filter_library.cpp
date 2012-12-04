/**
 * \file filter_library.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "test-env.h"
//#include <pvkernel/filter/PVFilterLibrary.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/filter/PVFieldsFilter.h>

#include <iostream>
#include <QString>

#include <pvkernel/core/picviz_assert.h>

typedef LIB_CLASS(PVFilter::PVFieldsFilter<PVFilter::one_to_many>) fields_filter_o2m_t;

int main()
{
	init_env();

	PVFilter::PVPluginsLoad::load_all_plugins();

	fields_filter_o2m_t::list_classes const lc1 = fields_filter_o2m_t::get().get_list();
	fields_filter_o2m_t::list_classes::const_iterator it;

	std::cout << "Splitters:" << std::endl;
	for (it = lc1.begin(); it != lc1.end(); it++) {
		std::cout << qPrintable(it.key()) << std::endl;
	}

	fields_filter_o2m_t::list_classes::const_iterator regexp_it = lc1.find("regexp");
	PV_ASSERT_VALID(regexp_it != lc1.end());

	PVFilter::PVFieldsSplitter_p regexp = regexp_it.value();
	PV_ASSERT_VALID(regexp.get() != nullptr);

	fields_filter_o2m_t::get().register_class("test", *regexp);
	fields_filter_o2m_t::list_classes const lc2 = fields_filter_o2m_t::get().get_list();
	fields_filter_o2m_t::list_classes::const_iterator it2;

	for (it2 = lc2.begin(); it2 != lc2.end(); it2++) {
		if (lc1.contains(it2.key())) {
			continue;
		} else if (it2.key() == "test") {
			continue;
		} else {
			std::cout << "where does the key '" << qPrintable(it2.key()) << "' come from" << std::endl;
			return 1;
		}
	}

	std::cout << "All fields filters:" << std::endl;
	for (it2 = lc2.begin(); it2 != lc2.end(); it2++) {
		std::cout << qPrintable(it2.key()) << std::endl;
	}

	return 0;
}
