/**
 * \file PVSourceDescription.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/rush/PVSourceDescription.h>

#include <boost/iterator/indirect_iterator.hpp>

#include <iostream>

bool PVRush::PVSourceDescription::operator==(const PVSourceDescription& other) const
{
	// FIXME: PVSourceCreator and PVFormat should have their own operator==

	return _inputs.size() == other._inputs.size() && //!\\ compare lists size before applying std::equal to avoid segfault !
			std::equal(boost::make_indirect_iterator(_inputs.begin()), boost::make_indirect_iterator(_inputs.end()), boost::make_indirect_iterator(other._inputs.begin())) &&
		   _source_creator_p->registered_name() == other._source_creator_p->registered_name() &&
		   _format.get_full_path() == other._format.get_full_path();
}
