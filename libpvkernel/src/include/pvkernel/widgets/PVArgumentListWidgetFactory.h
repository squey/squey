/**
 * \file PVArgumentListWidgetFactory.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVKERNEL_PVARGUMENTLISTWIDGETFACTORY_H
#define PVKERNEL_PVARGUMENTLISTWIDGETFACTORY_H

#include <pvbase/export.h>
#include <QItemEditorFactory>

namespace PVWidgets {

namespace PVArgumentListWidgetFactory {

LibKernelDecl QItemEditorFactory* create_core_widgets_factory();

}

}

#endif
