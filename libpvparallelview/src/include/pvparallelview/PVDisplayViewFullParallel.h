#ifndef PVDISPLAYS_PVDISPLAYVIEWFULLPARALLEL_H
#define PVDISPLAYS_PVDISPLAYVIEWFULLPARALLEL_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays {

class PVDisplayViewFullParallel: public PVDisplayViewIf
{
public:
	PVDisplayViewFullParallel();

public:
	QWidget* create_widget(Picviz::PVView* view, QWidget* parent) const override;

	CLASS_REGISTRABLE(PVDisplayViewFullParallel)
};

}

#endif
