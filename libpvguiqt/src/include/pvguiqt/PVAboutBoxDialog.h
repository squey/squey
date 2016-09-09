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

namespace PVGuiQt
{

namespace __impl
{
class GraphicsView;
}

class PVAboutBoxDialog : public QDialog
{
	friend class __impl::GraphicsView;

  public:
	PVAboutBoxDialog(QWidget* parent = 0);

  private:
	__impl::GraphicsView* _view3D;
	QHBoxLayout* _view3D_layout;
};

namespace __impl
{

class GraphicsView : public QGraphicsView
{
  public:
	GraphicsView(PVAboutBoxDialog* parent) : QGraphicsView(parent) {}

  protected:
	void resizeEvent(QResizeEvent* event);
};
}
}

#endif /* __PVGUIQT_PVABOUTBOXDIALOG_H__ */
