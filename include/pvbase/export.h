/**
 * \file export.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVBASE_EXPORT_H
#define PVBASE_EXPORT_H

#define LibExportTempl template class __declspec( dllexport )
#define LibImportTempl template class __declspec( dllimport )

#define LibExport __declspec( dllexport )
#define LibImport __declspec( dllimport )


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
#include "../../libpvkernel/src/include/pvkernel/export.h"
#include "../../libpicviz/src/include/picviz/export.h"

#endif	/* PVBASE_EXPORT_H */
