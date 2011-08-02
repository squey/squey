#ifndef PVRUSH_EXPORT_H
#define PVRUSH_EXPORT_H

#ifndef PVBASE_EXPORT_H
#error This file must not be included directly. Use pvbase/export.h instead.
#endif

#ifdef WIN32
 #ifdef pvrush_EXPORTS
  #define LibRushDeclExplicitTempl LibExportTempl
  #define LibRushDecl LibExport
 #else
  #define LibRushDeclExplicitTempl LibImportTempl
  #define LibRushDecl LibImport
 #endif
#else
 #define LibRushDeclExplicitTempl
 #define LibRushDecl
#endif

#endif
