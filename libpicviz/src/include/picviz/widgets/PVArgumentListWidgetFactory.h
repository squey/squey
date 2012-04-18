#ifndef PICVIZ_PVARGUMENTLISTWIDGETFACTORY_H
#define PICVIZ_PVARGUMENTLISTWIDGETFACTORY_H

#include <QItemEditorFactory>

namespace Picviz {
class PVView;
}

namespace PVWidgets {

namespace PVArgumentListWidgetFactory {

LibPicvizDecl QItemEditorFactory* create_layer_widget_factory(Picviz::PVView& view);
LibPicvizDecl QItemEditorFactory* create_mapping_plotting_widget_factory();

}

}

#endif
