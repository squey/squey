/**
 * \file PVSource.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvhive/PVHive.h>

#include <pvhive/waxes/picviz/PVSource.h>
#include <pvhive/waxes/picviz/PVView.h>

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

IMPL_WAX(Picviz::PVSource::process_from_source, src, args)
{
	call_object_default<Picviz::PVSource, FUNC(Picviz::PVSource::process_from_source)>(src, args);

	for (Picviz::PVView_sp& v: src->get_children<Picviz::PVView>()) {
		refresh_observers(&v->get_layer_stack_output_layer());
		refresh_observers(&v->get_pre_filter_layer());
		refresh_observers(&v->get_post_filter_layer());
		refresh_observers(&v->get_output_layer());
		refresh_observers(&v->get_real_output_selection());
	}
}

PVHIVE_CALL_OBJECT_BLOCK_END()
