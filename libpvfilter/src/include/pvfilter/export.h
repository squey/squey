#ifndef PVFILTER_EXPORT_H
#define PVFILTER_EXPORT_H

#ifndef PVCORE_EXPORT_H
#error This file must not be included directly. Use pvcore/export.h instead.
#endif

#ifdef WIN32
 #ifdef pvfilter_EXPORTS
  #define pvfilter_FilterLibraryDecl win32_FilterLibraryDeclExp
  #define LibFilterDecl LibExport
 #else
  #define pvfilter_FilterLibraryDecl win32_FilterLibraryDeclImp
  #define LibFilterDecl LibImport
 #endif
#else
 #define pvfilter_FilterLibraryDecl
 #define LibFilterDecl
#endif

#endif
