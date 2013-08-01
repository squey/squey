/**
 * \file PVFieldSplitterIronPortMail.cpp
 *
 * Copyright (C) Picviz Labs 2011-2013
 */

#include "PVFieldSplitterIronPortMail.h"
#include "PVFieldSplitterIronPortMailTag.h"

#include <pvkernel/core/PVBufferSlice.h>
#include <pvkernel/core/PVUnicodeString16.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVAxisTagsDec.h>

#include <QRegExp>

static const QString str_empty("");
static const QString str_unknown("unknown");

static const QString str_com_start("com-start");
static const QString str_com_finish("com-finish");
static const QString str_com_msgid("com-msgid");

static const QString str_msg_from("msg-from");
static const QString str_msg_to("msg-to");
static const QString str_msg_subject("msg-subject");
static const QString str_msg_size("msg-size");
static const QString str_msg_domain_keys("msg-domain-keys");
static const QString str_msg_dkim("msg-dkim");

static const QString str_dlvy_queued("dlvy-queued");
static const QString str_dlvy_start("dlvy-start");
static const QString str_dlvy_done("dlvy-done");
static const QString str_dlvy_reply("dlvy-reply");

typedef enum {
	COL_LOG_TYPE = 0,
	COL_LOG_TEXT_SHORT,

	COL_MSGID,

	COL_MSG_FROM,
	COL_MSG_TO,
	COL_MSG_SUBJECT,
	COL_MSG_SIZE,
	COL_MSG_DOMAIN_KEYS,
	COL_MSG_DKIM,

	COL_MID,
	COL_ICID,
	COL_RID,
	COL_DCID,

	COL_FIELDS_NUMBER
} col_type;

/******************************************************************************
 * PVFilter::PVCore::PVFieldSplitterIronPortMail::regexps_t
 *****************************************************************************/

PVFilter::PVFieldSplitterIronPortMail::regexps_t::regexps_t()
{
	text_short.setPattern("^MID \\d+ (.*)");

	com_start.setPattern      ("^Start MID (\\d+) ICID (\\d+)");
	com_finish.setPattern     ("^Message finished MID (\\d+) done");
	com_msgid.setPattern      ("^MID (\\d+) Message-ID '([^']*)'");

	msg_from.setPattern       ("^MID (\\d+) ICID (\\d+) From: <([^>]+)>");
	msg_to.setPattern         ("^MID (\\d+) ICID (\\d+) RID (\\d+) To: (.*)");
	msg_size.setPattern       ("^MID (\\d+) [^ ]+ (\\d+)");
	msg_subject.setPattern    ("^MID (\\d+) Subject \"([^\"]*)\"");
	msg_domain_keys.setPattern("^MID (\\d+) DomainKeys: (.*)");
	msg_dkim.setPattern       ("^MID (\\d+) DKIM: (.*)");

	dlvy_queued.setPattern    ("^MID (\\d+) queued for delivery");
	dlvy_start.setPattern     ("^Delivery start DCID (\\d+) MID (\\d+) to RID \\[(\\d+)\\]");
	dlvy_done.setPattern      ("^Message done DCID (\\d+) MID (\\d+) to RID \\[(\\d+)\\]");
	dlvy_reply.setPattern     ("^MID (\\d+) RID \\[(\\d+)\\] Response '.*'");
}

/******************************************************************************
 * PVFilter::PVCore::PVFieldSplitterIronPortMail::PVCore::PVFieldSplitterIronPortMail
 *****************************************************************************/

PVFilter::PVFieldSplitterIronPortMail::PVFieldSplitterIronPortMail() :
	PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER_NOPARAM(PVFilter::PVFieldSplitterIronPortMail);

	// Default tags position values (if the splitter is used outside a format)

	_col_log_type        = COL_LOG_TYPE;
	_col_log_text_short  = COL_LOG_TEXT_SHORT;

	_col_msgid           = COL_MSGID;

	_col_msg_from        = COL_MSG_FROM;
	_col_msg_to          = COL_MSG_TO;
	_col_msg_subject     = COL_MSG_SUBJECT;
	_col_msg_size        = COL_MSG_SIZE;
	_col_msg_domain_keys = COL_MSG_DOMAIN_KEYS;
	_col_msg_dkim        = COL_MSG_DKIM;

	_col_mid             = COL_MID;
	_col_icid            = COL_ICID;
	_col_rid             = COL_RID;
	_col_dcid            = COL_DCID;

	_ncols = COL_FIELDS_NUMBER;
}

void PVFilter::PVFieldSplitterIronPortMail::set_children_axes_tag(filter_child_axes_tag_t const& axes)
{
	PVFieldsBaseFilter::set_children_axes_tag(axes);

	_col_log_type        = axes.value(TAG_IPM_LOG_TYPE, -1);
	_col_log_text_short  = axes.value(TAG_IPM_LOG_TEXT_SHORT, -1);

	_col_msgid           = axes.value(TAG_IPM_MSGID, -1);

	_col_msg_from        = axes.value(TAG_IPM_MSG_FROM, -1);
	_col_msg_to          = axes.value(TAG_IPM_MSG_TO, -1);
	_col_msg_subject     = axes.value(TAG_IPM_MSG_SUBJECT, -1);
	_col_msg_size        = axes.value(TAG_IPM_MSG_SIZE, -1);
	_col_msg_domain_keys = axes.value(TAG_IPM_MSG_DOMAIN_KEYS, -1);
	_col_msg_dkim        = axes.value(TAG_IPM_MSG_DKIM, -1);

	_col_mid             = axes.value(TAG_IPM_MID, -1);
	_col_icid            = axes.value(TAG_IPM_ICID, -1);
	_col_rid             = axes.value(TAG_IPM_RID, -1);
	_col_dcid            = axes.value(TAG_IPM_DCID, -1);

	int num =  (_col_log_type != -1) + (_col_log_text_short != -1) + (_col_msgid != -1) +(_col_msg_from != -1) + (_col_msg_to != -1) + (_col_msg_subject != -1) + (_col_msg_size != -1) + (_col_msg_domain_keys != -1) + (_col_msg_dkim != -1) + (_col_mid != -1) + (_col_icid != -1) + (_col_rid != -1) + (_col_dcid != -1);

	if (num != COL_FIELDS_NUMBER) {
		PVLOG_WARN("(PVFieldSplitterIronPortMail::set_children_axes_tag) warning: Iron Port Mail splitter set but no tags have been found !\n");
	}

	_ncols = COL_FIELDS_NUMBER;

}

static bool set_field(int pos, PVCore::PVField** fields,
                      const uint16_t* str = nullptr,
                      const int str_begin = 0,
                      const int str_len = 0)
{
	if (pos == -1) {
		return false;
	}

	PVCore::PVField* new_f = fields[pos];
	const size_t str_end = str_begin + str_len;

	new_f->set_begin((char*) (str + str_begin));
	new_f->set_end((char*) (str + str_end));
	new_f->set_physical_end((char*) (str + str_end));

	return true;
}

static bool set_field(int pos, PVCore::PVField** fields,
                      const QString& str)
{
	if (pos == -1) {
		return false;
	}

	return set_field(pos, fields, str.utf16(), 0, str.size());
}

static bool set_field(int pos, PVCore::PVField** fields,
                      const uint16_t* str,
                      QRegExp &re, int p)
{
	if (pos == -1) {
		return false;
	}

	return set_field(pos, fields, str,
	                 re.pos(p), re.cap(p).size());
}

static bool set_short_text(int pos, PVCore::PVField** fields,
                           const uint16_t* str,
                           QRegExp& re, QString& text)
{
	if (re.indexIn(text) == -1) {
		return false;
	}

	return set_field(pos, fields, str,
	                 re.pos(1), re.cap(1).size());
}

/******************************************************************************
 * PVFilter::PVFieldSplitterIronPortMail::one_to_many
 *****************************************************************************/

PVCore::list_fields::size_type
PVFilter::PVFieldSplitterIronPortMail::one_to_many(PVCore::list_fields &l,
                                                   PVCore::list_fields::iterator it_ins,
                                                   PVCore::PVField &field)
{
	regexps_t& res = _regexps.local();

	QString text;
	PVCore::list_fields::size_type n = COL_FIELDS_NUMBER;

	const uint16_t* input_str = (const uint16_t*) field.begin();
	field.get_qstr(text);

	// Add the number of final fields and save their pointers
	PVCore::PVField *pf[COL_FIELDS_NUMBER];
	PVCore::PVField ftmp(*field.elt_parent());
	for (PVCol i = 0; i < _ncols; i++) {
		PVCore::list_fields::iterator it_new = l.insert(it_ins, ftmp);
		pf[i] = &(*it_new);
		set_field(i, pf, str_empty);
	}

	if (res.com_start.indexIn(text) != -1) {
		// COMMAND START
		set_field(_col_log_type, pf, str_com_start);

		set_field(_col_mid, pf, input_str, res.com_start, 1);
		set_field(_col_icid, pf, input_str, res.com_start, 2);
	} else if (res.com_finish.indexIn(text) != -1) {
		// COMMAND FINISH
		set_field(_col_log_type, pf, str_com_finish);

		set_field(_col_mid, pf, input_str, res.com_finish, 1);
	} else if (res.com_msgid.indexIn(text) != -1) {
		// MSG ID
		set_field(_col_log_type, pf, str_com_msgid);

		set_field(_col_mid, pf, input_str, res.com_msgid, 1);
		set_field(_col_msgid, pf, input_str, res.com_msgid, 2);
	} else if (res.msg_from.indexIn(text) != -1) {
		// FROM
		set_field(_col_log_type, pf, str_msg_from);
		set_short_text(_col_log_text_short, pf, input_str, res.text_short, text);

		set_field(_col_mid, pf, input_str, res.msg_from, 1);
		set_field(_col_icid, pf, input_str, res.msg_from, 2);
		set_field(_col_msg_from, pf, input_str, res.msg_from, 3);
	} else if (res.msg_to.indexIn(text) != -1) {
		// TO
		set_field(_col_log_type, pf, str_msg_to);
		set_short_text(_col_log_text_short, pf, input_str, res.text_short, text);

		set_field(_col_mid, pf, input_str, res.msg_to, 1);
		set_field(_col_icid, pf, input_str, res.msg_to, 2);
		set_field(_col_rid, pf, input_str, res.msg_to, 3);
		set_field(_col_msg_to, pf, input_str, res.msg_to, 4);
	} else if (res.msg_subject.indexIn(text) != -1) {
		// SUBJECT
		set_field(_col_log_type, pf, str_msg_subject);
		set_short_text(_col_log_text_short, pf, input_str, res.text_short, text);

		set_field(_col_mid, pf, input_str, res.msg_subject, 1);
		set_field(_col_msg_subject, pf, input_str, res.msg_subject, 2);
	} else if (res.msg_size.indexIn(text) != -1) {
		// SIZE
		set_field(_col_log_type, pf, str_msg_size);
		set_short_text(_col_log_text_short, pf, input_str, res.text_short, text);

		set_field(_col_mid, pf, input_str, res.msg_size, 1);
		set_field(_col_msg_size, pf, input_str, res.msg_size, 2);
	} else if (res.msg_domain_keys.indexIn(text) != -1) {
		// DOMAIN KEYS
		set_field(_col_log_type, pf, str_msg_domain_keys);
		set_short_text(_col_log_text_short, pf, input_str, res.text_short, text);

		set_field(_col_mid, pf, input_str, res.msg_domain_keys, 1);
		set_field(_col_msg_domain_keys, pf, input_str, res.msg_domain_keys, 2);
	} else if (res.msg_dkim.indexIn(text) != -1) {
		// DKIM
		set_field(_col_log_type, pf, str_msg_dkim);
		set_short_text(_col_log_text_short, pf, input_str, res.text_short, text);

		set_field(_col_mid, pf, input_str, res.msg_dkim, 1);
		set_field(_col_msg_dkim, pf, input_str, res.msg_dkim, 2);
	} else if (res.dlvy_queued.indexIn(text) != -1) {
		// DELIVERY QUEUED
		set_field(_col_log_type, pf, str_dlvy_queued);
		set_short_text(_col_log_text_short, pf, input_str, res.text_short, text);

		set_field(_col_mid, pf, input_str, res.dlvy_queued, 1);
	} else if (res.dlvy_reply.indexIn(text) != -1) {
		// DELIVERY REPLY
		set_field(_col_log_type, pf, str_dlvy_reply);
		set_short_text(_col_log_text_short, pf, input_str, res.text_short, text);

		set_field(_col_mid, pf, input_str, res.dlvy_reply, 1);
		set_field(_col_rid, pf, input_str, res.dlvy_reply, 2);
	} else if (res.dlvy_start.indexIn(text) != -1) {
		// DELIVERY START
		set_field(_col_log_type, pf, str_dlvy_start);

		set_field(_col_dcid, pf, input_str, res.dlvy_start, 1);
		set_field(_col_mid, pf, input_str, res.dlvy_start, 2);
		set_field(_col_rid, pf, input_str, res.dlvy_start, 3);
	} else if (res.dlvy_done.indexIn(text) != -1) {
		// DELIVERY DONE
		set_field(_col_log_type, pf, str_dlvy_done);

		set_field(_col_dcid, pf, input_str, res.dlvy_done, 1);
		set_field(_col_mid, pf, input_str, res.dlvy_done, 2);
		set_field(_col_rid, pf, input_str, res.dlvy_done, 3);
	} else {
		// UNKNOWN
		set_field(_col_log_type, pf, str_unknown);
	}

	return n;
}

IMPL_FILTER_NOPARAM(PVFilter::PVFieldSplitterIronPortMail)
