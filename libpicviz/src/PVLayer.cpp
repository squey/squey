//! \file PVLayer.cpp
//! $Id: PVLayer.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011


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

/******************************************************************************
 *
 * Picviz::PVLayer::set_lines_properties
 *
 *****************************************************************************/
void Picviz::PVLayer::set_lines_properties(const PVLinesProperties & lines_properties_)
{
	
}


/******************************************************************************
 *
 * Picviz::PVLayer::set_selection
 *
 *****************************************************************************/
void Picviz::PVLayer::set_selection(const PVSelection & selection_)
{

}






