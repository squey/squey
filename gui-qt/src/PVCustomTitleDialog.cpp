#include <PVCustomTitleDialog.h>
#include <QPainter>
#include <QStyle>
#include <QStyleOption>
#include <QRect>
#include <QStyleOptionTitleBar>

PVInspector::PVCustomTitleDialog::PVCustomTitleDialog(QWidget* parent, Qt::WindowFlags f):
	QDialog(parent, f | Qt::FramelessWindowHint)
{
}

void PVInspector::PVCustomTitleDialog::paintEvent(QPaintEvent* /*event*/)
{
	QPainter p(this);
	QStyle* style = this->style();
	QRect active_area = rect();
	int titlebar_height = 0;

	// Titlebar.
	QStyleOptionTitleBar t_opt;
	t_opt.initFrom(this);

	titlebar_height = style->pixelMetric(QStyle::PM_TitleBarHeight, &t_opt, this);

	t_opt.rect = QRect(0, 0, width(), titlebar_height);
	t_opt.titleBarState = windowState();
	t_opt.titleBarFlags = windowFlags();
	t_opt.text = t_opt.fontMetrics.elidedText(windowTitle(), Qt::ElideRight, t_opt.rect.width());
	style->drawComplexControl(QStyle::CC_TitleBar, &t_opt, &p, this);
	style->drawItemText(&p, t_opt.rect, Qt::AlignCenter, t_opt.palette, true, t_opt.text, QPalette::ToolTipText);

	// Background widget.
	active_area.setTopLeft(QPoint(0, titlebar_height));
	this->setContentsMargins(0, titlebar_height, 0, 0);

	QStyleOption w_opt;
	w_opt.initFrom(this);
	w_opt.rect = active_area;
	style->drawPrimitive(QStyle::PE_Widget, &w_opt, &p, this);
}
