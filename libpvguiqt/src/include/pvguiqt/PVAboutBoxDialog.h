/**
 * \file PVAboutBoxDialog.h
 *
 * Copyright (C) Picviz Labs 2012
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
	Q_OBJECT;
	friend class __impl::GraphicsView;
public:
	PVAboutBoxDialog(QWidget* parent = 0);

protected:
	void keyPressEvent(QKeyEvent* event);
private:
	 __impl::GraphicsView* _view3D;
	 bool _fullscreen;
	 QHBoxLayout* _view3D_layout;
};

namespace __impl
{

class GraphicsView : public QGraphicsView
{
public:
	GraphicsView(PVAboutBoxDialog* parent) : _parent(parent) {}

	void set_fullscreen(bool fullscreen = true);

protected:
    void resizeEvent(QResizeEvent *event);
    void keyPressEvent(QKeyEvent * event);

private:
    PVAboutBoxDialog* _parent;
};

}

}

#endif /* __PVGUIQT_PVABOUTBOXDIALOG_H__ */
