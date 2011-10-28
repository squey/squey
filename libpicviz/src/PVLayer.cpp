//! \file PVLayer.cpp
//! $Id: PVLayer.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <picviz/PVLayer.h>




/******************************************************************************
 *
 * Picviz::PVLayer::PVLayer
 *
 *****************************************************************************/
// Picviz::PVLayer::PVLayer(const QString & name_) :
// 	lines_properties(),
// 	name(name_),
// 	selection()
// {
// 	name.truncate(PICVIZ_LAYER_NAME_MAXLEN);
// }

/******************************************************************************
 *
 * Picviz::PVLayer::PVLayer
 *
 *****************************************************************************/
Picviz::PVLayer::PVLayer(const QString & name_, const PVSelection & sel_, const PVLinesProperties & lp_) :
	lines_properties(lp_),
	name(name_),
	selection(sel_)
{
	name.truncate(PICVIZ_LAYER_NAME_MAXLEN);
	locked = false;
	visible = true;
}


/******************************************************************************
 *
 * Picviz::PVLayer::operator=
 *
 *****************************************************************************/
Picviz::PVLayer & Picviz::PVLayer::operator=(const PVLayer & rhs)
{
	// We check for self assignment
	if (this == &rhs) {
		return *this;
	}

	lines_properties = rhs.get_lines_properties();
	selection = rhs.get_selection();

	return *this;
}

/******************************************************************************
 *
 * Picviz::PVLayer::A2B_copy_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Picviz::PVLayer::A2B_copy_restricted_by_selection_and_nelts(PVLayer &b, PVSelection const& selection, pvrow nelts)
{
	get_lines_properties().A2B_copy_restricted_by_selection_and_nelts(b.get_lines_properties(), selection, nelts);
	b.get_selection() &= get_selection();
}

/******************************************************************************
 *
 * Picviz::PVLayer::reset_to_empty_and_default_color
 *
 *****************************************************************************/
void Picviz::PVLayer::reset_to_empty_and_default_color()
{
	lines_properties.reset_to_default_color();
	selection.select_none();
}

/******************************************************************************
 *
 * Picviz::PVLayer::reset_to_full_and_default_color
 *
 *****************************************************************************/
void Picviz::PVLayer::reset_to_full_and_default_color()
{
	lines_properties.reset_to_default_color();
	selection.select_all();
}

void Picviz::PVLayer::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.object("selection", selection);
	so.object("lp", lines_properties);
	so.attribute("name", name);
	so.attribute("visible", visible);
	so.attribute("index", index);
	so.attribute("locked", locked);
}


void Picviz::PVLayer::load_from_file(QString const& path)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchiveZip(path, PVCore::PVSerializeArchive::read, PICVIZ_ARCHIVES_VERSION));
	ar->get_root()->object("layer", *this);
	ar->finish();
#endif
}

void Picviz::PVLayer::save_to_file(QString const& path)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchiveZip(path, PVCore::PVSerializeArchive::write, PICVIZ_ARCHIVES_VERSION));
	ar->get_root()->object("layer", *this);
	ar->finish();
#endif
}
