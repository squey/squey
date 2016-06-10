/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvhive/PVHive.h>

#include <pvhive/waxes/inendi/PVSource.h>
#include <pvhive/waxes/inendi/PVView.h>

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

IMPL_WAX(Inendi::PVSource::set_axis_hovered, src, args)
{
	call_object_default<Inendi::PVSource, FUNC(Inendi::PVSource::set_axis_hovered)>(src, args);

	refresh_observers(&src->axis_hovered());
}

IMPL_WAX(Inendi::PVSource::set_section_hovered, src, args)
{
	call_object_default<Inendi::PVSource, FUNC(Inendi::PVSource::set_section_hovered)>(src, args);

	refresh_observers(&src->section_hovered());
}

IMPL_WAX(Inendi::PVSource::set_section_clicked, src, args)
{
	call_object_default<Inendi::PVSource, FUNC(Inendi::PVSource::set_section_clicked)>(src, args);

	refresh_observers(&src->section_clicked());
}

IMPL_WAX(Inendi::PVSource::set_axis_clicked, src, args)
{
	call_object_default<Inendi::PVSource, FUNC(Inendi::PVSource::set_axis_clicked)>(src, args);

	refresh_observers(&src->axis_clicked());
}

PVHIVE_CALL_OBJECT_BLOCK_END()
