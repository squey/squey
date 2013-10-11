#ifndef PVDISPLAYS_PVDISPLAYVIEWZOOMEDPARALLEL_H
#define PVDISPLAYS_PVDISPLAYVIEWZOOMEDPARALLEL_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays {

class PVDisplayViewZoomedParallel: public PVDisplayViewAxisIf
{
public:
	PVDisplayViewZoomedParallel();

public:
	QWidget* create_widget(Picviz::PVView* view, PVCol axis_comb, QWidget* parent) const override;
	QIcon toolbar_icon() const override;
	QString widget_title(Picviz::PVView* view, PVCol axis_comb) const override;
	QString axis_menu_name(Picviz::PVView const* view, PVCol axis_comb) const override;

	CLASS_REGISTRABLE(PVDisplayViewZoomedParallel)
};

}

#endif // PVDISPLAYS_PVDISPLAYVIEWZOOMEDPARALLEL_H
