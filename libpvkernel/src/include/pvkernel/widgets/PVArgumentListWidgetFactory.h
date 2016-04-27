/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVKERNEL_PVARGUMENTLISTWIDGETFACTORY_H
#define PVKERNEL_PVARGUMENTLISTWIDGETFACTORY_H

#include <pvbase/export.h>
#include <QItemEditorFactory>

namespace PVWidgets
{

namespace PVArgumentListWidgetFactory
{

QItemEditorFactory* create_core_widgets_factory();
}
}

#endif
