/**
 * \file PVSourceDescription.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/rush/PVSourceDescription.h>

#include <boost/iterator/indirect_iterator.hpp>

#include <QFile>

bool PVRush::PVSourceDescription::operator==(const PVSourceDescription& other) const
{
	// FIXME: PVSourceCreator and PVFormat should have their own operator==

	return _inputs.size() == other._inputs.size() && //!\\ compare lists size before applying std::equal to avoid segfault !
			std::equal(boost::make_indirect_iterator(_inputs.begin()), boost::make_indirect_iterator(_inputs.end()), boost::make_indirect_iterator(other._inputs.begin())) &&
		   _source_creator_p->registered_name() == other._source_creator_p->registered_name() &&
		   _format.get_full_path() == other._format.get_full_path();
}

bool PVRush::PVSourceDescription::is_valid()
{
	bool is_valid = true;
	for (const PVInputDescription_p& input_desc : _inputs) {
		is_valid &= QFile::exists(input_desc->human_name());
	}
	is_valid &= QFile::exists(_format.get_full_path());
	return is_valid;
}
