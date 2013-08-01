/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2013
 */

#include <pvkernel/core/PVClassLibrary.h>

#include "PVFieldSplitterIronPortMail.h"
#include "PVFieldSplitterIronPortMailTag.h"
#include "PVFieldSplitterIronPortMailParamWidget.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("ironport_mail", PVFilter::PVFieldSplitterIronPortMail);
	REGISTER_CLASS_AS("splitter_ironport_mail",
	                  PVFilter::PVFieldSplitterIronPortMail,
	                  PVFilter::PVFieldsFilterReg);

	DECLARE_TAG(TAG_IPM_LOG_TYPE       , TAG_IPM_LOG_TYPE_DESC       , PVFilter::PVFieldSplitterIronPortMail);
	DECLARE_TAG(TAG_IPM_LOG_TEXT_SHORT , TAG_IPM_LOG_TEXT_SHORT_DESC , PVFilter::PVFieldSplitterIronPortMail);

	DECLARE_TAG(TAG_IPM_MSGID          , TAG_IPM_MSGID_DESC          , PVFilter::PVFieldSplitterIronPortMail);

	DECLARE_TAG(TAG_IPM_MSG_FROM       , TAG_IPM_MSG_FROM_DESC       , PVFilter::PVFieldSplitterIronPortMail);
	DECLARE_TAG(TAG_IPM_MSG_TO         , TAG_IPM_MSG_TO_DESC         , PVFilter::PVFieldSplitterIronPortMail);
	DECLARE_TAG(TAG_IPM_MSG_SUBJECT    , TAG_IPM_MSG_SUBJECT_DESC    , PVFilter::PVFieldSplitterIronPortMail);
	DECLARE_TAG(TAG_IPM_MSG_SIZE       , TAG_IPM_MSG_SIZE_DESC       , PVFilter::PVFieldSplitterIronPortMail);
	DECLARE_TAG(TAG_IPM_MSG_DOMAIN_KEYS, TAG_IPM_MSG_DOMAIN_KEYS_DESC, PVFilter::PVFieldSplitterIronPortMail);
	DECLARE_TAG(TAG_IPM_MSG_DKIM       , TAG_IPM_MSG_DKIM_DESC       , PVFilter::PVFieldSplitterIronPortMail);

	DECLARE_TAG(TAG_IPM_MID            , TAG_IPM_MID_DESC            , PVFilter::PVFieldSplitterIronPortMail);
	DECLARE_TAG(TAG_IPM_ICID           , TAG_IPM_ICID_DESC           , PVFilter::PVFieldSplitterIronPortMail);
	DECLARE_TAG(TAG_IPM_RID            , TAG_IPM_RID_DESC            , PVFilter::PVFieldSplitterIronPortMail);
	DECLARE_TAG(TAG_IPM_DCID           , TAG_IPM_DCID_DESC           , PVFilter::PVFieldSplitterIronPortMail);

	REGISTER_CLASS("ironport_mail", PVFilter::PVFieldSplitterIronPortMailParamWidget);
}
