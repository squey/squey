#ifndef PVGL_EXPORT_H
#define PVGL_EXPORT_H

#ifndef PVBASE_EXPORT_H
#error This file must not be included directly. Use pvbase/export.h instead.
#endif

#ifdef WIN32
 #ifdef pvgl_EXPORTS
  #define pvgl_FilterLibraryDecl win32_FilterLibraryDeclExp
  #define LibGLDecl LibExport
 #else
  #define pvgl_FilterLibraryDecl win32_FilterLibraryDeclImp
  #define LibGLDecl LibImport
 #endif
#else
 #define pvgl_FilterLibraryDecl
 #define LibGLDecl
#endif

#endif
