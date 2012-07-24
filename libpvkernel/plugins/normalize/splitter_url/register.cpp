/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldSplitterURL.h"

#include <pvkernel/rush/PVAxisTagsDec.h>

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("url", PVFilter::PVFieldSplitterURL);
	REGISTER_CLASS_AS("splitter_url", PVFilter::PVFieldSplitterURL, PVFilter::PVFieldsFilterReg);
	DECLARE_TAG(PVAXIS_TAG_HOST, PVAXIS_TAG_HOST_DESC, PVFilter::PVFieldSplitterURL);
	DECLARE_TAG(PVAXIS_TAG_SUBDOMAIN, PVAXIS_TAG_SUBDOMAIN_DESC, PVFilter::PVFieldSplitterURL);
	DECLARE_TAG(PVAXIS_TAG_DOMAIN, PVAXIS_TAG_DOMAIN_DESC, PVFilter::PVFieldSplitterURL);
	DECLARE_TAG(PVAXIS_TAG_PROTOCOL, PVAXIS_TAG_PROTOCOL_DESC, PVFilter::PVFieldSplitterURL);
	DECLARE_TAG(PVAXIS_TAG_PORT, PVAXIS_TAG_PORT_DESC, PVFilter::PVFieldSplitterURL);
	DECLARE_TAG(PVAXIS_TAG_URL, PVAXIS_TAG_URL_DESC, PVFilter::PVFieldSplitterURL);
	DECLARE_TAG(PVAXIS_TAG_TLD, PVAXIS_TAG_TLD_DESC, PVFilter::PVFieldSplitterURL);
	DECLARE_TAG(PVAXIS_TAG_URL_VARIABLES, PVAXIS_TAG_URL_VARIABLES_DESC, PVFilter::PVFieldSplitterURL);
	DECLARE_TAG(PVAXIS_TAG_URL_FRAGMENT, PVAXIS_TAG_URL_VARIABLES_DESC, PVFilter::PVFieldSplitterURL);
	DECLARE_TAG(PVAXIS_TAG_URL_CREDENTIALS, PVAXIS_TAG_URL_VARIABLES_DESC, PVFilter::PVFieldSplitterURL);
}
