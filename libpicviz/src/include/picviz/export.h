#ifndef PICVIZ_EXPORT_H
#define PICVIZ_EXPORT_H

#ifndef PVCORE_EXPORT_H
#error This file must not be included directly. Use pvcore/export.h instead.
#endif

#ifdef WIN32
 #ifdef picviz_EXPORTS
  #define LibPicvizDeclExplicitTempl LibExportTempl
  #define LibPicvizDecl LibExport
 #else
  #define LibPicvizDeclExplicitTempl LibImportTempl
  #define LibPicvizDecl LibImport
 #endif
#else
 #define LibPicvizDeclExplicitTempl
 #define LibPicvizDecl
#endif

#endif
