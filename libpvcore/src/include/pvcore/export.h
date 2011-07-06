/*
 * $Id: export.h 3101 2011-06-10 08:57:30Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVCORE_EXPORT_H
#define PVCORE_EXPORT_H

#ifdef WIN32
#define LibExport __declspec( dllexport )
#else
#define LibExport
#endif

#ifdef WIN32
#define LibCPPExport extern "C" __declspec( dllexport )
#else
#define LibCPPExport extern "C"
#endif

#ifdef WIN32			/* FIXME: This is totally weird! If I don't export plugin symbols like this, I have unknown symbols under Linux */
#define PluginExport __declspec( dllexport )
#else
#define PluginExport extern "C"
#endif

#define win32_FilterLibraryDeclExp template class __declspec( dllexport )
#define win32_FilterLibraryDeclImp extern template class __declspec( dllimport )

#ifdef WIN32
 #ifdef pvfilter_EXPORTS
  #define pvfilter_FilterLibraryDecl win32_FilterLibraryDeclExp
 #else
  #define pvfilter_FilterLibraryDecl win32_FilterLibraryDeclImp
 #endif
#else
 #define pvfilter_FilterLibraryDecl LibExport
#endif

#ifdef WIN32
 #ifdef pvrush_EXPORTS
  #define pvrush_FilterLibraryDecl win32_FilterLibraryDeclExp
 #else
  #define pvrush_FilterLibraryDecl win32_FilterLibraryDeclImp
 #endif
#else
 #define pvrush_FilterLibraryDecl LibExport
#endif

#ifdef WIN32
 #ifdef picviz_EXPORTS
  #define picviz_FilterLibraryDecl win32_FilterLibraryDeclExp
 #else
  #define picviz_FilterLibraryDecl win32_FilterLibraryDeclImp
 #endif
#else
 #define picviz_FilterLibraryDecl LibExport
#endif

#endif	/* PVCORE_EXPORT_H */
