#ifndef PVFILTER_EXPORT_H
#define PVFILTER_EXPORT_H

#ifndef PVCORE_EXPORT_H
#error This file must not be included directly. Use pvcore/export.h instead.
#endif

#ifdef WIN32
 #ifdef pvfilter_EXPORTS
  #define LibFilterDeclExplicitTempl LibExportTempl
  #define LibFilterDecl LibExport
 #else
  #define LibFilterDeclExplicitTempl LibImportTempl
  #define LibFilterDecl LibImport
 #endif
#else
 #define LibFilterDeclExplicitTempl
 #define LibFilterDecl
#endif


#endif
