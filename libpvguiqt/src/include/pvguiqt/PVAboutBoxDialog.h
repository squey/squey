/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVGUIQT_PVABOUTBOXDIALOG_H__
#define __PVGUIQT_PVABOUTBOXDIALOG_H__

#include <QDialog>
#include <QHBoxLayout>
#include <QGraphicsView>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QWidget>

class QTabWidget;

namespace PVGuiQt
{

namespace __impl
{
class GraphicsView;
} // namespace __impl

class PVAboutBoxDialog : public QDialog
{
  public:
	enum Tab { SOFTWARE, CHANGELOG, REFERENCE_MANUAL, OPEN_SOURCE_SOFTWARE };

	friend class __impl::GraphicsView;

  public:
	explicit PVAboutBoxDialog(Tab = SOFTWARE, QWidget* parent = nullptr);
	void select_tab(Tab);

  private:
	__impl::GraphicsView* _view3D;
	QHBoxLayout* _view3D_layout;

	QTabWidget* _tab_widget;
	QWidget* _changelog_tab;
	QWidget* _reference_manual_tab;
};

namespace __impl
{

class GraphicsView : public QGraphicsView
{
  public:
	explicit GraphicsView(PVAboutBoxDialog* parent) : QGraphicsView(parent) {}

  protected:
	void resizeEvent(QResizeEvent* event) override;
};
} // namespace __impl
} // namespace PVGuiQt

#endif /* __PVGUIQT_PVABOUTBOXDIALOG_H__ */
