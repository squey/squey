/**
 * \file PVDisplayViewListing.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVDISPLAYS_PVDISPLAYVIEWLISTING_H
#define PVDISPLAYS_PVDISPLAYVIEWLISTING_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays {

class PVDisplayViewListing: public PVDisplayViewIf
{
public:
	PVDisplayViewListing();

public:
	QWidget* create_widget(Picviz::PVView* view, QWidget* parent) const override;
	QIcon toolbar_icon() const override;
	QString widget_title(Picviz::PVView* view) const override;

	CLASS_REGISTRABLE(PVDisplayViewListing)
};

}

#endif
