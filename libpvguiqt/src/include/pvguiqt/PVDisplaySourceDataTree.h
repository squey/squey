#ifndef PVDISPLAYS_PVDISPLAYSOURCELISTING_H
#define PVDISPLAYS_PVDISPLAYSOURCELISTING_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays {

class PVDisplaySourceDataTree: public PVDisplaySourceIf
{
public:
	PVDisplaySourceDataTree();

public:
	QWidget* create_widget(Picviz::PVView* view, QWidget* parent) const override;

	CLASS_REGISTRABLE(PVDisplaySourceDataTree)
};

}

#endif
