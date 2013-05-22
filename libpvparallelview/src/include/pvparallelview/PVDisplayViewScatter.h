/**
 * \file PVDisplayViewScatter.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVDISPLAYVIEWSCATTER_H__
#define __PVDISPLAYVIEWSCATTER_H__

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays {

class PVDisplayViewScatter: public PVDisplayViewZoneIf
{
public:
	PVDisplayViewScatter();

public:
	QWidget* create_widget(Picviz::PVView* view, PVCol axis_comb, QWidget* parent) const override;
	QIcon toolbar_icon() const override;
	QString widget_title(Picviz::PVView* view, PVCol axis_comb) const override;
	QString axis_menu_name(Picviz::PVView const* view, PVCol axis_comb) const override;

	CLASS_REGISTRABLE(PVDisplayViewScatter)
};

}

#endif // __PVDISPLAYVIEWSCATTER_H__
