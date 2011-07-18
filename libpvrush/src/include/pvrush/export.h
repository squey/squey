#ifndef PVRUSH_EXPORT_H
#define PVRUSH_EXPORT_H

#ifndef PVCORE_EXPORT_H
#error This file must not be included directly. Use pvcore/export.h instead.
#endif

#ifdef WIN32
 #ifdef pvrush_EXPORTS
  #define pvrush_FilterLibraryDecl win32_FilterLibraryDeclExp
  #define LibRushDecl LibExport
 #else
  #define pvrush_FilterLibraryDecl win32_FilterLibraryDeclImp
  #define LibRushDecl LibImport
 #endif
#else
 #define pvrush_FilterLibraryDecl
 #define LibRushDecl
#endif

#endif
