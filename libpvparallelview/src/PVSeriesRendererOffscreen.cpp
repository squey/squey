//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvparallelview/PVSeriesRendererOffscreen.h>

#include <pvkernel/core/PVConfig.h>
#include <QSettings>
#include <tuple>
#include <QDebug>
#include <EGL/egl.h>
#include <GL/gl.h>

namespace PVParallelView
{

#define EGLCHECK(func)                                                                             \
	[&](auto&&... args) {                                                                          \
		if (func(args...) == EGL_FALSE) {                                                          \
			qDebug() << #func << "fails:" << eglGetError();                                        \
		}                                                                                          \
	}

#define CONFIGATTR(attr) get_config_attr(display, conf, attr)
#define PRINT_CONFIGATTR(attr) qDebug() << #attr << CONFIGATTR(attr);

static EGLint get_config_attr(EGLDisplay display, EGLConfig config, EGLint attr)
{
	EGLint value = 0;
	EGLCHECK(eglGetConfigAttrib)(display, config, attr, &value);
	return value;
};

static void debug_config(EGLDisplay display, EGLConfig conf)
{
	PRINT_CONFIGATTR(EGL_CONFIG_ID);
	PRINT_CONFIGATTR(EGL_SURFACE_TYPE);
	PRINT_CONFIGATTR(EGL_BUFFER_SIZE);
	PRINT_CONFIGATTR(EGL_RED_SIZE);
	PRINT_CONFIGATTR(EGL_GREEN_SIZE);
	PRINT_CONFIGATTR(EGL_BLUE_SIZE);
	PRINT_CONFIGATTR(EGL_LUMINANCE_SIZE);
	PRINT_CONFIGATTR(EGL_ALPHA_SIZE);
	PRINT_CONFIGATTR(EGL_ALPHA_MASK_SIZE);
	PRINT_CONFIGATTR(EGL_BIND_TO_TEXTURE_RGB);
	PRINT_CONFIGATTR(EGL_BIND_TO_TEXTURE_RGBA);
	PRINT_CONFIGATTR(EGL_COLOR_BUFFER_TYPE);
	PRINT_CONFIGATTR(EGL_CONFIG_CAVEAT);
	PRINT_CONFIGATTR(EGL_CONFORMANT);
	PRINT_CONFIGATTR(EGL_DEPTH_SIZE);
	PRINT_CONFIGATTR(EGL_LEVEL);
	PRINT_CONFIGATTR(EGL_MATCH_NATIVE_PIXMAP);
	PRINT_CONFIGATTR(EGL_MAX_PBUFFER_WIDTH);
	PRINT_CONFIGATTR(EGL_MAX_PBUFFER_HEIGHT);
	PRINT_CONFIGATTR(EGL_MAX_PBUFFER_PIXELS);
	PRINT_CONFIGATTR(EGL_MAX_SWAP_INTERVAL);
	PRINT_CONFIGATTR(EGL_MIN_SWAP_INTERVAL);
	PRINT_CONFIGATTR(EGL_NATIVE_RENDERABLE);
	PRINT_CONFIGATTR(EGL_NATIVE_VISUAL_ID);
	PRINT_CONFIGATTR(EGL_NATIVE_VISUAL_TYPE);
	PRINT_CONFIGATTR(EGL_RENDERABLE_TYPE);
	PRINT_CONFIGATTR(EGL_SAMPLE_BUFFERS);
	PRINT_CONFIGATTR(EGL_SAMPLES);
	PRINT_CONFIGATTR(EGL_STENCIL_SIZE);
	PRINT_CONFIGATTR(EGL_SURFACE_TYPE);
	PRINT_CONFIGATTR(EGL_TRANSPARENT_TYPE);
	PRINT_CONFIGATTR(EGL_TRANSPARENT_RED_VALUE);
	PRINT_CONFIGATTR(EGL_TRANSPARENT_GREEN_VALUE);
	PRINT_CONFIGATTR(EGL_TRANSPARENT_BLUE_VALUE);
}

using EGLDeviceEXT = void *;

static std::vector<EGLDeviceEXT> get_devices()
{
	EGLBoolean (*eglQueryDevicesEXT)(EGLint max_devices, EGLDeviceEXT * devices,
	                                 EGLint * num_devices) = nullptr;
	eglQueryDevicesEXT =
	    reinterpret_cast<decltype(eglQueryDevicesEXT)>(eglGetProcAddress("eglQueryDevicesEXT"));
	if (eglQueryDevicesEXT == nullptr) {
		qDebug() << "eglQueryDevicesEXT not available";
		return {};
	}

	EGLint num_devices = 0;
	EGLCHECK(eglQueryDevicesEXT)(0, nullptr, &num_devices);
	std::vector<EGLDeviceEXT> devices(num_devices);
	EGLCHECK(eglQueryDevicesEXT)(num_devices, devices.data(), &num_devices);
	return devices;
}

struct EGL {
	bool acquire(EGLDisplay display)
	{
		if (auto it = m_displays.find(display); it != end(m_displays)) {
			++it->second;
			return true;
		}
		EGLint major = 0, minor = 0;
		if (eglInitialize(display, &major, &minor) == EGL_FALSE) {
			qDebug() << "eglInitialize fails:" << eglGetError();
			return false;
		}
		m_displays.emplace(display, 1);
		return true;
	}

	void release(EGLDisplay display)
	{
		auto it = m_displays.find(display);
		if (it == end(m_displays)) {
			qDebug() << "EGLDisplay was terminated before last usage";
			return;
		}
		--it->second;
		if (it->second == 0) {
			EGLCHECK(eglTerminate)(display);
			m_displays.erase(it);
		}
	}

  private:
	std::unordered_map<EGLDisplay, size_t> m_displays;
};

static EGL g_EGL_instance;

static EGLDisplay get_display(EGLDeviceEXT device)
{
#define EGL_PLATFORM_DEVICE_EXT 0x313F
	EGLAttrib const display_attrib_list[] = {EGL_NONE};
	EGLDisplay display =
	    eglGetPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, device, display_attrib_list);

	if (display != EGL_NO_DISPLAY) {
		if (not g_EGL_instance.acquire(display)) {
			return EGL_NO_DISPLAY;
		}
	}
	return display;
}

static EGLContext test_config(EGLDisplay display, EGLConfig config)
{
	EGLint const context_attrs[]{
	    EGL_CONTEXT_MAJOR_VERSION, PVSeriesRendererOffscreen::OpenGLES_version_major,
	    EGL_CONTEXT_MINOR_VERSION, PVSeriesRendererOffscreen::OpenGLES_version_minor, EGL_NONE};

	EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrs);
	if (context == EGL_NO_CONTEXT) {
		qDebug() << __func__ << "EGL_CONFIG_ID" << get_config_attr(display, config, EGL_CONFIG_ID)
		         << "eglCreateContext fails:" << eglGetError();
		return EGL_NO_CONTEXT;
	}

	if (eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context) == EGL_FALSE) {
		qDebug() << __func__ << "EGL_CONFIG_ID" << get_config_attr(display, config, EGL_CONFIG_ID)
		         << "eglMakeCurrent fails:" << eglGetError();
		eglDestroyContext(display, context);
		return EGL_NO_CONTEXT;
	} else {
		return context;
	}
}

static EGLContext choose_config(EGLDisplay display)
{
	EGLint const attribs[]{EGL_SURFACE_TYPE,
	                       EGL_PBUFFER_BIT,
	                       EGL_RENDERABLE_TYPE,
	                       EGL_OPENGL_ES2_BIT,
	                       EGL_BUFFER_SIZE,
	                       24,
	                       EGL_NONE};
	EGLint numconf = 0;
	EGLCHECK(eglChooseConfig)(display, attribs, nullptr, 0, &numconf);
	if (numconf == 0) {
		qDebug() << __func__ << "eglChooseConfig could not find any matching config";
		return EGL_NO_CONTEXT;
	}
	std::vector<EGLConfig> matching_configs(numconf);
	EGLCHECK(eglChooseConfig)(display, attribs, matching_configs.data(), numconf, &numconf);
	EGLCHECK(eglBindAPI)(EGL_OPENGL_ES_API);
	for (EGLConfig conf : matching_configs) {
		if (EGLContext context = test_config(display, conf); context != EGL_NO_CONTEXT) {
			return context;
		}
	}
	qDebug() << __func__ << "Found" << numconf << "matching EGLConfig but none could work";
	return EGL_NO_CONTEXT;
}

bool egl_support()
{
	return PVSeriesRendererOffscreen::capability();
}

QString egl_vendor()
{
	if (not egl_support()) {
		return {};
	}
	auto display = get_display(get_devices()[0]);
	g_EGL_instance.acquire(display);
	auto str = "EGL " + QString(eglQueryString(display, EGL_VERSION)) + " " +
	           QString(eglQueryString(display, EGL_VENDOR));
	g_EGL_instance.release(display);
	return str;
}

QString opengl_version()
{
	if (not egl_support()) {
		return {};
	}
	auto display = get_display(get_devices()[0]);
	g_EGL_instance.acquire(display);
	EGLContext context = choose_config(display);
	eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context);
	auto str = QString(reinterpret_cast<const char*>(glGetString(GL_VERSION)));
	eglDestroyContext(display, context);
	g_EGL_instance.release(display);
	return str;
}

PVSeriesRendererOffscreen::PVSeriesRendererOffscreen(Squey::PVRangeSubSampler const& rss)
    : PVSeriesAbstractRenderer(rss), QOffscreenSurface(), _gl_renderer(rss)
{
	std::vector<EGLDeviceEXT> devices = get_devices();

	if (devices.empty()) {
		qDebug() << "PVSeriesRendererOffscreen: no EGL platform device available";
		return;
	}

	EGLDisplay display = get_display(devices[0]);
	// EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if (display == EGL_NO_DISPLAY or not g_EGL_instance.acquire(display)) {
		qDebug() << "PVSeriesRendererOffscreen: could not init EGLDisplay";
		return;
	};
	qDebug() << "\nEGL_CLIENT_APIS:" << eglQueryString(display, EGL_CLIENT_APIS)
	         << "\nEGL_EXTENSIONS:" << eglQueryString(display, EGL_EXTENSIONS)
	         << "\nEGL_VENDOR:" << eglQueryString(display, EGL_VENDOR)
	         << "\nEGL_VERSION:" << eglQueryString(display, EGL_VERSION);

	QSurfaceFormat format;
	format.setRenderableType(QSurfaceFormat::OpenGLES);
	format.setVersion(OpenGLES_version_major, OpenGLES_version_minor);
	format.setProfile(QSurfaceFormat::CoreProfile);
	QOffscreenSurface::setFormat(format);
	QOffscreenSurface::create();
	_gl_renderer.setFormat(QOffscreenSurface::format());
	qDebug() << "Could init QOffscreenSurface:" << isValid();
}

bool PVSeriesRendererOffscreen::capability()
{
	// temporarily disable accelerated backends
	return false;

	static const bool s_offscreenopengl_capable = [] {
		if (PVCore::PVConfig::get().config().value("backend_opencl/force_cpu", false).toBool()) {
			qDebug() << "backend_opencl/force_cpu is set to true in user config";
			return false;
		}

		qDebug() << "EGL_EXTENSIONS:" << eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);

		std::vector<EGLDeviceEXT> devices = get_devices();
		qDebug() << "PVSeriesRendererOffscreen::capability()"
		         << "found EGL platform devices:" << devices.size();
		if (devices.empty()) {
			return false;
		}

		EGLDisplay display = get_display(devices[0]);
		if (display == EGL_NO_DISPLAY or not g_EGL_instance.acquire(display)) {
			qDebug() << "PVSeriesRendererOffscreen::capability()"
			         << "could not init EGLDisplay";
			return false;
		};
		qDebug() << "EGL_CLIENT_APIS:" << eglQueryString(display, EGL_CLIENT_APIS)
		         << "\nEGL_EXTENSIONS:" << eglQueryString(display, EGL_EXTENSIONS)
		         << "\nEGL_VENDOR:" << eglQueryString(display, EGL_VENDOR)
		         << "\nEGL_VERSION:" << eglQueryString(display, EGL_VERSION);
		
		if (auto vendor = eglQueryString(display, EGL_VENDOR); vendor != std::string("NVIDIA")) {
			qDebug() << "Only supports NVIDIA, got " << vendor;
			g_EGL_instance.release(display);
			return false;
		}

		EGLContext context = choose_config(display);

		if (context == EGL_NO_CONTEXT) {
			qDebug() << "PVSeriesRendererOffscreen::capability()"
			         << "could not init EGLContext";
			g_EGL_instance.release(display);
			return false;
		}

		QSurfaceFormat format;
		format.setRenderableType(QSurfaceFormat::OpenGLES);
		format.setVersion(OpenGLES_version_major, OpenGLES_version_minor);
		format.setProfile(QSurfaceFormat::CoreProfile);

		QOffscreenSurface offsc;
		offsc.setFormat(format);
		offsc.create();
		if (not offsc.isValid()) {
			qDebug() << "Imposible to create QOffscreenSurface";
			g_EGL_instance.release(display);
			return false;
		}
		QOpenGLContext qogl;
		qogl.setFormat(offsc.format());
		if (not qogl.create()) {
			qDebug() << "Could not create a QOpenGLContext out of the QOffscreenSurface";
		} else if (auto expected_version =
		               qMakePair(OpenGLES_version_major, OpenGLES_version_minor);
		           qogl.format().version() < expected_version) {
			qDebug() << "Expecting" << expected_version
			         << "but QOffscreenSurface could only deliver " << qogl.format().version();
		} else if (not qogl.makeCurrent(&offsc)) {
			qDebug() << "Could not make QOpenGLContext current on QOffscreenSurface";
		} else {
			qogl.doneCurrent();
			g_EGL_instance.release(display);
			return true;
		}
		g_EGL_instance.release(display);
		return false;
	}();
	return s_offscreenopengl_capable;
}

} // namespace PVParallelView
