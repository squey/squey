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

	CLASS_REGISTRABLE(PVDisplayViewListing)
};

}

#endif
