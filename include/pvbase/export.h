/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
