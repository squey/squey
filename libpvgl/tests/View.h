#include <QGLWidget>
#include "Point.h"

#define NB_LINES 1048576*5

class View : public QGLWidget
{
	Q_OBJECT

	public:
	View(QWidget *parent) : QGLWidget(parent) { buffer_size = 0;}
	void set_buffer(Point *p, int size);
	 QSize sizeHint() const { return QSize(1024,1024); };
	virtual QSize minimumSizeHint() const { return QSize(300,300); }

	private:
	virtual void initializeGL()
	{
		glClearColor(0,0,0,0);
		//glDisable(GL_DEPTH_TEST);
		glEnable(GL_DEPTH_TEST);
	}

	virtual void resizeGL(int w, int h)
	{
		glViewport(0,0,(GLint)w, (GLint)h);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.0,1024,0,1024,0,1000000000);

		glMatrixMode(GL_MODELVIEW);
	}


	void paintGL();

	Point* buffer;
	int buffer_size;
};
