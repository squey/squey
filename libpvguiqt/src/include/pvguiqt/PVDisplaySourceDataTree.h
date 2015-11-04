/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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
	QWidget* create_widget(Picviz::PVSource* src, QWidget* parent) const override;
	QIcon toolbar_icon() const override;

	QString widget_title(Picviz::PVSource*) const override { return QString("Data tree"); } 

	CLASS_REGISTRABLE(PVDisplaySourceDataTree)
};

}

#endif
