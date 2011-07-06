//! \file PVAxis.h
//! $Id: PVAxis.h 2590 2011-05-07 15:43:12Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVAXIS_H
#define PICVIZ_PVAXIS_H

#include <pvcore/general.h>

#include <picviz/PVColor.h>

namespace Picviz {

/**
 * \class PVAxis
 */
class LibExport PVAxis {
public:
	float absciss;
	/* PVCol column_index; Needed in PVAxesCombination, we shall remove it */
	Picviz::PVColor titlecolor;
	Picviz::PVColor color;
	bool is_expandable;
	bool is_expanded;
	bool is_key;
	QString name;
	QString type;
	QString modemapping;
	QString modeplotting;
	float thickness;

	/**
	 * Constructor
	 */
	PVAxis();

	/**
	 * Destructor
	 */
	~PVAxis();
	
	/**
	 * Gets the name of the axis
	 *
	 * @return The name of the axis.
	 */
	QString get_name();


	/**
	 * Sets the name of the axis
	 *
	 * @param name_ The new name to be set
	 *
	 */
	void set_name(const QString &name_);
	

};
}

#endif	/* PICVIZ_PVAXIS_H */
