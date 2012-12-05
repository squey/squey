/**
 * \file shared_ptr.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include "test-env.h"

using namespace PVFilter;

//FIXME: what this program is supposed to test?

std::list<PVFilter::PVFieldsFilterReg_p> container;
PVFieldsBaseFilter_f get_f()
{
	PVFilter::PVFieldsSplitter::p_type filter = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("regexp");
	PVFilter::PVFieldsFilterReg_p filter_clone = filter->clone<PVFilter::PVFieldsFilterReg>();
	container.push_back(filter_clone);
	return filter_clone->f();
}

int main()
{
	init_env();
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVFieldsBaseFilter_f f = get_f();
	const char* test = "salut";

	return 0;
}
