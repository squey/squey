#ifndef PVFIELDSPLITTERCSVPARAMWIDGET_H
#define PVFIELDSPLITTERCSVPARAMWIDGET_H

#include <pvcore/general.h>
#include <pvfilter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

namespace PVFilter {

class PVFieldSplitterCSVParamWidget: public PVFieldsFilterParamWidget<one_to_many>
{
public:
	PVFieldSplitterCSVParamWidget();
public:
	QWidget* get_widget();

	CLASS_REGISTRABLE(PVFieldSplitterCSVParamWidget)
};

}

#endif
