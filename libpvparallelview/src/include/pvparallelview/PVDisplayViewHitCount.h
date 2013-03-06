
#ifndef PVDISPLAYS_PVDISPLAYVIEWHITCOUNT_H
#define PVDISPLAYS_PVDISPLAYVIEWHITCOUNT_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays {

class PVDisplayViewHitCount: public PVDisplayViewAxisIf
{
public:
	PVDisplayViewHitCount();

public:
	QWidget* create_widget(Picviz::PVView* view, PVCol axis_comb, QWidget* parent) const override;
	QIcon toolbar_icon() const override;
	QString widget_title(Picviz::PVView* view, PVCol axis_comb) const override;
	QString axis_menu_name(Picviz::PVView const* view, PVCol axis_comb) const override;

	CLASS_REGISTRABLE(PVDisplayViewHitCount)
};

}

#endif // PVDISPLAYS_PVDISPLAYVIEWHITCOUNT_H
