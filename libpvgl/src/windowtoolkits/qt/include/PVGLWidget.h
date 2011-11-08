#ifdef USE_WTK_QT

#ifndef PVGL_WTK_QT_WIDGET_H
#define PVGL_WTK_QT_WIDGET_H

#include <QGLWidget>
#include <QWidget>

namespace PVGL {
namespace WTKQt {

class PVGLWidget: public QGLWidget
{
	Q_OBJECT
public:
	PVGLWidget(int id, QWidget* parent = 0);

public:
	int id() const { return _win_id; }

protected:
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();

private:
	int _win_id;
};

}
}

#endif

#endif
