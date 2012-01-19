#include <QGLWidget>
#include "Point.h"
#include <iostream>

//#define NB_LINES 1048576
//#define NB_LINES 128*1000000
#define NB_LINES 128*10000


class View : public QGLWidget
{
	Q_OBJECT

	public:
	View(QWidget *parent);
	void set_buffer(Point *p, int size);
	 QSize sizeHint() const { return QSize(1024,1024); };
	virtual QSize minimumSizeHint() const { return QSize(300,300); }

	private:
		
		void compute_collision_table();
		
		void initializeGL()
		{
			glClearColor(1,0,0,0);
			glDisable(GL_DEPTH_TEST);
			//glEnable(GL_DEPTH_TEST);
		}

		void reset_collision_table();

		void resizeGL(int w, int h)
		{
			std::cout << "Format du View: " << w << "x" << h << std::endl;
			glViewport(0,0, (GLint)w, (GLint)h);

			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(0.0, 1024, 0, 1024, 0, 1000000000);

			glMatrixMode(GL_MODELVIEW);
			//paintGL();
		}


		void paintGL();

		Point* buffer;
		int buffer_size;
		int ugly_collision_table[1024*1024];
};
