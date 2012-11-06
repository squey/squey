#ifndef PVDISPLAYS_PVDISPLAYVIEWAXISZOMMED_H
#define PVDISPLAYS_PVDISPLAYVIEWAXISZOMMED_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays {

class PVDisplayViewAxisZoomed: public PVDisplayViewAxisIf
{
public:
	PVDisplayViewAxisZoomed();

public:
	QWidget* create_widget(Picviz::PVView* view, PVCol axis_comb, QWidget* parent) const override;

	CLASS_REGISTRABLE(PVDisplayViewAxisZoomed)
};

}

#endif
