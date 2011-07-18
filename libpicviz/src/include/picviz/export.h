#ifndef PICVIZ_EXPORT_H
#define PICVIZ_EXPORT_H

#ifndef PVCORE_EXPORT_H
#error This file must not be included directly. Use pvcore/export.h instead.
#endif

#ifdef WIN32
 #ifdef picviz_EXPORTS
  #define picviz_FilterLibraryDecl win32_FilterLibraryDeclExp
  #define LibPicvizDecl LibExport
 #else
  #define picviz_FilterLibraryDecl win32_FilterLibraryDeclImp
  #define LibPicvizDecl LibImport
 #endif
#else
 #define picviz_PicvizLibraryDecl
 #define LibPicvizDecl
#endif

#endif
