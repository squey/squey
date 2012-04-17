#ifndef SLVIEW_H
#define SLVIEW_H

#include <QGLWidget>
#include <iostream>

struct PVRGB
{
	union {
		uint32_t int_v;
		struct {
			uint8_t r;
			uint8_t g;
			uint8_t b;
			uint8_t a;
		} s;
	};
};

// Tc == type of the coordinates
template <class Tc>
class SLView: public QGLWidget
{
public:
	SLView(QWidget *parent): QGLWidget(parent) { _colors = NULL; };
	void set_points(std::vector<Tc> const& pts) { _pts = &pts; };
	void set_colors(std::vector<PVRGB> const& colors) { _colors = &colors; };
	void set_ortho(Tc w, Tc h) { _ortho_w = w; _ortho_h = h; }
	void set_size(int w, int h) { _w = w; _h = h; }
	QSize sizeHint() const { return QSize(_w,_h); };
	virtual QSize minimumSizeHint() const { return QSize(_w, _h); }

private:
	void initializeGL()
	{
		glClearColor(0,0,0,0);
	}

	void resizeGL(int w, int h)
	{
		glViewport(0,0, (GLint)w, (GLint)h);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.0, _ortho_w, 0.0, _ortho_h, 0, 1000000000);

		glMatrixMode(GL_MODELVIEW);
	}

protected:
	int _w;
	int _h;
	Tc _ortho_w;
	Tc _ortho_h;
	std::vector<Tc> const* _pts;
	std::vector<PVRGB> const* _colors;
};

#endif
