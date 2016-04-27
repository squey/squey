/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVARGUMENTLISTWIDGETFACTORY_H
#define INENDI_PVARGUMENTLISTWIDGETFACTORY_H

#include <QItemEditorFactory>

namespace Inendi
{
class PVView;
}

namespace PVWidgets
{

namespace PVArgumentListWidgetFactory
{

QItemEditorFactory* create_layer_widget_factory(Inendi::PVView const& view);
QItemEditorFactory* create_mapping_plotting_widget_factory();
}
}

#endif
