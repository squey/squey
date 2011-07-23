#include "PVFieldSplitterRegexpParamWidget.h"
#include "PVFieldSplitterRegexp.h"
#include <pvfilter/PVFieldsFilter.h>


#include <QLabel>
#include <QVBoxLayout>

#include <QSpacerItem>
#include <QPushButton>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexpParamWidget::PVFieldSplitterRegexpParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterRegexpParamWidget::PVFieldSplitterRegexpParamWidget() :
	PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterRegexp()))
{
}

