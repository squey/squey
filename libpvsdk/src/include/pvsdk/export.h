/**
 * \file export.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVSDK_EXPORT_H
#define PVSDK_EXPORT_H

#ifndef PVBASE_EXPORT_H
#error This file must not be included directly. Use pvbase/export.h instead.
#endif

#ifdef WIN32
 #ifdef pvsdk_EXPORTS
  #define pvsdk_FilterLibraryDecl win32_FilterLibraryDeclExp
  #define LibSDKDecl LibExport
 #else
  #define pvsdk_FilterLibraryDecl win32_FilterLibraryDeclImp
  #define LibSDKDecl LibImport
 #endif
#else
 #define pvsdk_FilterLibraryDecl
 #define LibSDKDecl
#endif

#endif	/* PVSDL */
