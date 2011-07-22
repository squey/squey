#ifndef PVFIELDSPLITTERCSVPARAMWIDGET_H
#define PVFIELDSPLITTERCSVPARAMWIDGET_H

#include <pvcore/general.h>
#include <pvfilter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

namespace PVFilter {

class PVFieldSplitterCSVParamWidget: public PVFieldsSplitterParamWidget
{
public:
	PVFieldSplitterCSVParamWidget();
public:
	QWidget* get_param_widget();

	CLASS_REGISTRABLE(PVFieldSplitterCSVParamWidget)
};

}

#endif
