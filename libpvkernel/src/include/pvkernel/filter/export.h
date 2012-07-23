/**
 * \file export.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVFILTER_EXPORT_H
#define PVFILTER_EXPORT_H

#ifndef PVBASE_EXPORT_H
#error This file must not be included directly. Use pvbase/export.h instead.
#endif

#ifdef WIN32
 #ifdef pvfilter_EXPORTS
  #define LibKernelDeclExplicitTempl LibExportTempl
  #define LibKernelDecl LibExport
 #else
  #define LibKernelDeclExplicitTempl LibImportTempl
  #define LibKernelDecl LibImport
 #endif
#else
 #define LibKernelDeclExplicitTempl
 #define LibKernelDecl
#endif


#endif
