/**
 * \file PVSourceDescription.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/rush/PVSourceDescription.h>

bool compare_input_descr(PVRush::PVInputDescription_p input_desc1, PVRush::PVInputDescription_p input_desc2)
{
	return (*input_desc1.get() == *input_desc2.get());
}

bool PVRush::PVSourceDescription::operator==(const PVSourceDescription& other) const
{
	// FIXME: PVSourceCreator and PVFormat should have their own operator==
	return std::equal(_inputs.begin(), _inputs.end(),  other._inputs.begin(), compare_input_descr) &&
		   _source_creator_p->registered_name() == other._source_creator_p->registered_name() &&
		   _format.get_full_path() == other._format.get_full_path();
}
