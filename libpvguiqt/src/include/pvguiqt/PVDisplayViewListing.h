/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
	QWidget* create_widget(Inendi::PVView* view, QWidget* parent) const override;
	QIcon toolbar_icon() const override;
	QString widget_title(Inendi::PVView* view) const override;

	CLASS_REGISTRABLE(PVDisplayViewListing)
};

}

#endif
