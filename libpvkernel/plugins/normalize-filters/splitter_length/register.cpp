/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

#include <pvbase/export.h>

#include <pvkernel/core/PVClassLibrary.h>

#include "PVFieldSplitterLength.h"
#include "PVFieldSplitterLengthParamWidget.h"

LibCPPExport void register_class()
{
	REGISTER_CLASS("length", PVFilter::PVFieldSplitterLength);
	REGISTER_CLASS_AS("splitter_length", PVFilter::PVFieldSplitterLength,
	                  PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("length", PVFilter::PVFieldSplitterLengthParamWidget);
}
