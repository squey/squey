#ifndef PICVIZ_EXPORT_H
#define PICVIZ_EXPORT_H

#ifndef PVBASE_EXPORT_H
#error This file must not be included directly. Use pvbase/export.h instead.
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
