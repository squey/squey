#ifndef __PVGUIQT_PVIMPORTWORKFLOWTABBAR_H__
#define __PVGUIQT_PVIMPORTWORKFLOWTABBAR_H__

#include <QTabBar>

namespace PVGuiQt
{

class PVImportWorkflowTabBar : public QTabBar
{
	static constexpr size_t ARROW_WIDTH = 10;
	static constexpr size_t PEN_WIDTH = 2;
	static constexpr uint64_t COLOR = 0x259ae9;

public:
	PVImportWorkflowTabBar(QWidget* parent = nullptr) : QTabBar(parent){}

	QSize tabSizeHint(int index) const override;
	QSize minimumTabSizeHint(int index) const override;

	void paintEvent(QPaintEvent* event) override;

};

} // namepace PVGuiQt

#endif // __PVGUIQT_PVIMPORTWORKFLOWTABBAR_H__