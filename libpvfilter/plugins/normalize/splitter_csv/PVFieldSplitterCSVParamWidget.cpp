#include "PVFieldSplitterCSVParamWidget.h"
#include "PVFieldSplitterCSV.h"
#include <pvfilter/PVFieldsFilter.h>

PVFilter::PVFieldSplitterCSVParamWidget::PVFieldSplitterCSVParamWidget() :
	PVFieldsFilterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterCSV()))
{
}

QWidget* PVFilter::PVFieldSplitterCSVParamWidget::get_widget()
{
	return NULL;
}
