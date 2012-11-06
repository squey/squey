#ifndef PVDISPLAYS_PVDISPLAYVIEWAXESCOMBINATION_H
#define PVDISPLAYS_PVDISPLAYVIEWAXESCOMBINATION_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays {

class PVDisplayViewAxesCombination: public PVDisplayViewIf
{
public:
	PVDisplayViewAxesCombination();

public:
	QWidget* create_widget(Picviz::PVView* view, QWidget* parent) const override;

	CLASS_REGISTRABLE(PVDisplayViewAxesCombination)
};

}

#endif
