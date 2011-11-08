#ifdef USE_WTK_QT

#ifndef PVGL_WTK_QT_COMMON_H
#define PVGL_WTK_QT_COMMON_H

#include <map>

namespace PVGL {
namespace WTKQt {

class PVGLWidget;

typedef std::map<int, PVGLWidget*> map_glwindows;

}
}

#endif

#endif
