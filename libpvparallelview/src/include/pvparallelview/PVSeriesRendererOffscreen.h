/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef _PVSERIESRENDEREROFFSCREEN_H_
#define _PVSERIESRENDEREROFFSCREEN_H_

#include <pvparallelview/PVSeriesRendererOpenGL.h>

#include <QOffscreenSurface>
#include <QDebug>
#include <EGL/egl.h>
#include <QtPlatformHeaders/QEGLNativeContext>
//#include <QtPlatformHeaders/QtPlatformHeaders>

namespace PVParallelView
{

class PVSeriesRendererOffscreen : public PVSeriesAbstractRenderer, public QOffscreenSurface
{
  public:
	virtual ~PVSeriesRendererOffscreen() = default;

	void set_background_color(QColor const& bgcol) override
	{
		_gl_renderer.set_background_color(bgcol);
	}
	void resize(QSize const& size) override { _gl_renderer.resize(size); }
	QPixmap grab() override { return _gl_renderer.grab(); }
	void set_draw_mode(PVSeriesView::DrawMode mode) override { _gl_renderer.set_draw_mode(mode); }

	PVSeriesRendererOffscreen(Inendi::PVRangeSubSampler const& rss)
	    : PVSeriesAbstractRenderer(rss), QOffscreenSurface(), _gl_renderer(rss)
	{
		QSurfaceFormat format;
		format.setRenderableType(QSurfaceFormat::OpenGLES);
		format.setVersion(3, 2);
		format.setProfile(QSurfaceFormat::CoreProfile);
		setFormat(format);

#define EGLCHECK(func)                                                                             \
	[&](auto&&... args) {                                                                          \
		if (func(args...) == EGL_FALSE) {                                                          \
			qDebug() << #func << "fails:" << eglGetError();                                        \
		}                                                                                          \
	}

		qDebug() << "\nEGL_EXTENSIONS:" << eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);

		typedef void* EGLDeviceEXT;
		EGLBoolean (*eglQueryDevicesEXT)(EGLint max_devices, EGLDeviceEXT * devices,
		                                 EGLint * num_devices) = nullptr;
		eglQueryDevicesEXT =
		    reinterpret_cast<decltype(eglQueryDevicesEXT)>(eglGetProcAddress("eglQueryDevicesEXT"));
		if (eglQueryDevicesEXT == nullptr) {
			qDebug() << "eglQueryDevicesEXT not available";
		}

		EGLint num_devices = 0;
		EGLCHECK(eglQueryDevicesEXT)(0, nullptr, &num_devices);
		std::vector<EGLDeviceEXT> devices(num_devices);
		EGLCHECK(eglQueryDevicesEXT)(num_devices, devices.data(), &num_devices);

		qDebug() << "num_devices:" << num_devices;

#define EGL_PLATFORM_DEVICE_EXT 0x313F
		EGLAttrib const display_attrib_list[] = {EGL_NONE};
		EGLDisplay display =
		    eglGetPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devices[0], display_attrib_list);

		// EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
		if (display == EGL_NO_DISPLAY) {
			qDebug() << "EGL_NO_DISPLAY";
		}
		EGLint major = 0, minor = 0;
		if (eglInitialize(display, &major, &minor) == EGL_FALSE) {
			qDebug() << "eglInitialize fails:" << eglGetError();
		}
		qDebug() << "\nEGL_CLIENT_APIS:" << eglQueryString(display, EGL_CLIENT_APIS)
		         << "\nEGL_EXTENSIONS:" << eglQueryString(display, EGL_EXTENSIONS)
		         << "\nEGL_VENDOR:" << eglQueryString(display, EGL_VENDOR)
		         << "\nEGL_VERSION:" << eglQueryString(display, EGL_VERSION);
		EGLint config_num = 0;
		if (eglGetConfigs(display, nullptr, 0, &config_num) == EGL_FALSE) {
			qDebug() << "eglGetConfigs fails:" << eglGetError();
		}
		qDebug() << "eglGetConfigs success:" << config_num;
		std::vector<EGLConfig> egl_configs(config_num);
		if (eglGetConfigs(display, egl_configs.data(), config_num, &config_num) == EGL_FALSE) {
			qDebug() << "eglGetConfigs fails:" << eglGetError();
		}

		auto get_config_attr = [](EGLDisplay display, EGLConfig config, EGLint attr) {
			EGLint value = 0;
			if (eglGetConfigAttrib(display, config, attr, &value) == EGL_FALSE) {
				qDebug() << "eglGetConfigAttrib fails:" << eglGetError();
			}
			return value;
		};

#define CONFIGATTR(attr) get_config_attr(display, conf, attr)
#define PRINT_CONFIGATTR(attr) qDebug() << #attr << CONFIGATTR(attr);

		for (auto& conf : egl_configs) {
			PRINT_CONFIGATTR(EGL_CONFIG_ID);
			PRINT_CONFIGATTR(EGL_MAX_PBUFFER_PIXELS);
			PRINT_CONFIGATTR(EGL_CONFIG_CAVEAT);
			PRINT_CONFIGATTR(EGL_CONFORMANT);
			PRINT_CONFIGATTR(EGL_SURFACE_TYPE);
			qDebug() << "Support of PBUFFER"
			         << bool(get_config_attr(display, conf, EGL_SURFACE_TYPE) & EGL_PBUFFER_BIT);
			qDebug() << "========================================";
		}
		EGLConfig chosen_config = [&]() {
			EGLint const attribs[]{EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE,
			                       EGL_OPENGL_ES2_BIT, EGL_NONE};
			EGLConfig conf = 0;
			EGLint numconf = 0;
			if (eglChooseConfig(display, attribs, &conf, 1, &numconf) == EGL_FALSE) {
				qDebug() << "eglChooseConfig fails:" << eglGetError();
			}
			if (numconf == 0) {
				qDebug() << "eglChooseConfig could not find any matching config";
			}
			return conf;
		}();
		// EGLSurface surface = eglCreatePbufferSurface(display, chosen_config, NULL);
		// if (surface == EGL_NO_SURFACE) {
		// 	qDebug() << "eglCreatePbufferSurface fails: " << eglGetError();
		// }
		EGLCHECK(eglBindAPI)(EGL_OPENGL_ES_API);

		EGLint const context_attrs[]{EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 2,
		                             EGL_NONE};

		EGLContext context =
		    eglCreateContext(display, chosen_config, EGL_NO_CONTEXT, context_attrs);
		if (context == EGL_NO_CONTEXT) {
			qDebug() << "eglCreateContext fails:" << eglGetError();
		}

		// auto* xcb = QXcbIntegration::instance();

		QOffscreenSurface::setNativeHandle(new QEGLNativeContext(context, display));
		QOffscreenSurface::create();
		_gl_renderer.setFormat(QOffscreenSurface::format());
		_gl_renderer.setNativeContext(QVariant::fromValue(QEGLNativeContext(context, display)));
		qDebug() << "Could init QOffscreenSurface:" << isValid();
	}

	static bool capability()
	{
		static const bool s_offscreenopengl_capable = [] {
			QSurfaceFormat format;
			format.setRenderableType(QSurfaceFormat::OpenGLES);
			format.setVersion(3, 2);
			format.setProfile(QSurfaceFormat::CoreProfile);

			QOffscreenSurface offsc;
			offsc.setFormat(format);
			offsc.create();
			if (not offsc.isValid()) {
				qDebug() << "Imposible to create QOffscreenSurface";
				QOffscreenSurface offsc_crash;
				offsc_crash.create();
				if (not offsc_crash.isValid()) {
					qDebug() << "Absolutely impossible to create any QOffscreenSurface";
				}
				return true; // rollback!
			}
			QOpenGLContext qogl;
			qogl.setFormat(offsc.format());
			if (not qogl.create()) {
				qDebug() << "Could not create a QOpenGLContext out of the QOffscreenSurface";
			} else if (qogl.format().version() < qMakePair(3, 2)) {
				qDebug() << "Expecting 3.2+ but QOffscreenSurface could only deliver "
				         << qogl.format().version();
			} else if (not qogl.makeCurrent(&offsc)) {
				qDebug() << "Could not make QOpenGLContext current on QOffscreenSurface";
			} else {
				qogl.doneCurrent();
				return true;
			}
			return true; // rollback!
		}();
		return s_offscreenopengl_capable;
	}

	template <class... Args>
	static auto capability(Args&&... args)
	{
		return PVSeriesRendererOpenGL::capability(std::forward<Args>(args)...);
	}

  protected:
	void on_show_series() override { _gl_renderer.show_series(std::move(_series_draw_order)); }

	PVSeriesRendererOpenGL _gl_renderer;
};

} // namespace PVParallelView

#endif // _PVSERIESRENDEREROFFSCREEN_H_
