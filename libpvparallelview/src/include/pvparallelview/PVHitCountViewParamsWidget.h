
#ifndef PVPARALLELVIEW_PVHITCOUNTVIEWPARAMSWIDGET_H
#define PVPARALLELVIEW_PVHITCOUNTVIEWPARAMSWIDGET_H

#include <pvkernel/widgets/PVConfigPopupWidget.h>

class QCheckBox;

namespace PVParallelView
{

class PVHitCountView;

class PVHitCountViewParamsWidget: public PVWidgets::PVConfigPopupWidget
{
public:
	PVHitCountViewParamsWidget(PVHitCountView* parent);

public:
	void update_widgets();

private:
	PVHitCountView* parent_hcv();

private:
	QCheckBox* _cb_autofit;
	QCheckBox* _cb_use_log_color;
};

}

#endif // PVPARALLELVIEW_PVHITCOUNTVIEWPARAMSWIDGET_H
