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


#define LibCPPExport extern "C"
#define PluginExport extern "C"

#endif	/* PVBASE_EXPORT_H */
