/**
 * \file export.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef SAD_EXPORT_H
#define SAD_EXPORT_H


#ifdef WIN32
#define LibCPPExport extern "C" __declspec( dllexport )
#else
#define LibCPPExport extern "C"
#endif


#ifdef WIN32
 #ifdef sad_EXPORTS
  #define LibSadDeclExplicitTempl LibExportTempl
  #define LibSadDecl LibExport
 #else
  #define LibSadDeclExplicitTempl LibImportTempl
  #define LibSadDecl LibImport
 #endif
#else
 #define LibSadDeclExplicitTempl
 #define LibSadDecl
#endif

#endif
