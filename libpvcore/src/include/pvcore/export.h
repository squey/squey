/*
 * $Id: export.h 3101 2011-06-10 08:57:30Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVCORE_EXPORT_H
#define PVCORE_EXPORT_H

#define LibExportTempl template class __declspec( dllexport )
#define LibImportTempl template class __declspec( dllimport )

#define LibExport __declspec( dllexport )
#define LibImport __declspec( dllimport )

#ifdef WIN32
 #ifdef pvcore_EXPORTS
  #define LibCoreDeclExplicitTempl LibExportTempl
  #define LibCoreDecl LibExport
 #else
  #define LibCoreDeclExplicitTempl LibImportTempl
  #define LibCoreDecl LibImport
 #endif
#else
 #define LibCoreDeclExplicitTempl
 #define LibCoreDecl
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

// Decls for other libraries
#include "../../../../libpvfilter/src/include/pvfilter/export.h"
#include "../../../../libpvrush/src/include/pvrush/export.h"
#include "../../../../libpicviz/src/include/picviz/export.h"
#include "../../../../libpvgl/src/include/pvgl/export.h"

#endif	/* PVCORE_EXPORT_H */
