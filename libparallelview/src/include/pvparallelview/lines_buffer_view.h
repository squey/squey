#ifndef SLVIEW_H
#define SLVIEW_H

#include <QGLWidget>
#include <QGLBuffer>
#include <iostream>
#include <pvkernel/core/picviz_bench.h>

#include <GL/gl.h>
#include <GL/glext.h>

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

class LBView : public QGLWidget
{
public:
	LBView(QWidget *parent, int width, int height): QGLWidget(parent)
	{
		makeCurrent();

		_width = width;
		_height = height;

		// std::cerr << "making a fake texture" << std::endl;
		// unsigned *img = new unsigned [_width * _height];
		// for (int i = 0; i < _width * _height; ++i) {
		// 	img[i] = 0xFF0000FF;
		// }

		std::cerr << "enabling GL_TEXTURE_2D" << std::endl;
		glEnable(GL_TEXTURE_2D);
		std::cout << "GLerror: " << glGetError() << std::endl;

		std::cerr << "generating texture ID" << std::endl;
		glGenTextures  (1, &_texid);
		std::cout << "GLerror: " << glGetError() << std::endl;
		std::cerr << "using Texture Id: " << _texid << std::endl;


		std::cerr << "binding texture ID" << std::endl;
		glBindTexture(GL_TEXTURE_2D, _texid);
		std::cout << "GLerror: " << std::hex << glGetError() << std::endl;

		// std::cerr << "loading image of size " << _width << "×" << _height << std::endl;
		// glTexImage2D(GL_TEXTURE_2D, 0,
		//              GL_RGBA, _width, _height, 0,
		//              GL_RGBA, GL_UNSIGNED_BYTE, img);
		// std::cout << "GLerror: " << glGetError() << std::endl;

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

		_pixel_buffer = new QGLBuffer(QGLBuffer::PixelUnpackBuffer);
		_pixel_buffer->setUsagePattern(QGLBuffer::DynamicCopy);
		_pixel_buffer->create();
		if (_pixel_buffer->isCreated() == false) {
			std::cerr << "error while creating QGLbuffer" << std::endl;
		}
		if (_pixel_buffer->bind() == false) {
			std::cerr << "error while binding QGLbuffer" << std::endl;
		} else {
			std::cerr << "QGLbuffer is binded: " << _pixel_buffer->bufferId() << std::endl;
		}
		std::cerr << "allocating a QGLbuffer" << std::endl;

		_pixel_buffer->allocate(width*height*sizeof(unsigned));
		if (_pixel_buffer->size() < 0) {
			std::cerr << "error while allocating QGLbuffer" << std::endl;
		}
	}

	void set_ortho(int w, int h) { _ortho_w = w; _ortho_h = h; }
	void set_size(int w, int h) { _w = w; _h = h; }
	QSize sizeHint() const { return QSize(_w,_h); };
	virtual QSize minimumSizeHint() const { return QSize(_w, _h); }

	GLuint get_buffer_id()
	{
		if (_pixel_buffer)
			return _pixel_buffer->bufferId();
		else
			return 0;
	}

private:
	void initializeGL()
	{
		glClearColor(0,.25,0,0);
		glEnable(GL_TEXTURE_2D);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	void resizeGL(int w, int h)
	{
		glViewport(0,0, (GLint)w, (GLint)h);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(-2.0, 2., -2.0, 2., 0, 1000000);

		glMatrixMode(GL_MODELVIEW);
	}

	void paintGL()
	{
		std::cout << "::paintGL()" << std::endl;
		// We clear the Buffers
		glClear(GL_COLOR_BUFFER_BIT);
		glColor3f(1,1,1);

		glDisable(GL_DEPTH_TEST);

		glEnable (GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, _texid);
		_pixel_buffer->bind();
		glTexImage2D(GL_TEXTURE_2D, 0,
		             GL_RGBA, _width, _height, 0,
		             GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		std::cout << "GLerror: " << glGetError() << std::endl;

		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f);
		glVertex2f(-1.0f, -1.0f);
		glTexCoord2f(1.0f, 0.0f);
		glVertex2f(1.0f, -1.0f);
		glTexCoord2f(1.0f, 1.0f);
		glVertex2f(1.0f, 1.0f);
		glTexCoord2f(0.0f, 1.0f);
		glVertex2f(-1.0f, 1.0f);
		glEnd();

		glFinish();
	}

protected:
	int _width, _height;
	int _w;
	int _h;
	int _ortho_w;
	int _ortho_h;
	QGLBuffer *_pixel_buffer;
	GLuint _texid;
};

#endif
