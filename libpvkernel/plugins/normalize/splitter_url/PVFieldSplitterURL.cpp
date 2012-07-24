/**
 * \file PVFieldSplitterURL.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

// Handles urls like this:
// lintranet.beijaflore.com:443
// clients1.google.com:443

#include "PVFieldSplitterURL.h"
#include <pvkernel/core/PVBufferSlice.h>
#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVAxisTagsDec.h>

#include <furl/decode.h>

#include <QUrl>

static const uint16_t empty_str = 0;
static const PVCore::PVUnicodeString g_str_http((const PVCore::PVUnicodeString::utf_char*) "h\0t\0t\0p\0", 4);
static const PVCore::PVUnicodeString g_str_https((const PVCore::PVUnicodeString::utf_char*)"h\0t\0t\0p\0s\0", 5);
static const PVCore::PVUnicodeString g_str_ftp((const PVCore::PVUnicodeString::utf_char*)"f\0t\0p\0", 3);
static const PVCore::PVUnicodeString::utf_char g_port_80[] = {'8', '0'};
static const PVCore::PVUnicodeString::utf_char g_port_443[] = {'4','4','3'};
static const PVCore::PVUnicodeString::utf_char g_port_21[] = {'2', '1'};

#define URL_NUMBER_FIELDS_CREATED 10

/******************************************************************************
 *
 * PVFilter::PVCore::PVFieldSplitterURL::PVCore::PVFieldSplitterURL
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterURL::PVFieldSplitterURL() :
	PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER_NOPARAM(PVFilter::PVFieldSplitterURL);

	// Default tags position values (if the splitter is used outside a format)
	_col_proto = 0;
	_col_subdomain = 1;
	_col_host = 2;
	_col_domain = 3;
	_col_tld = 4;
	_col_port = 5;
	_col_url = 6;
	_col_variable = 7;
	_col_fragment = 8;
	_col_credentials = 9;
	_ncols = 10;
}

void PVFilter::PVFieldSplitterURL::set_children_axes_tag(filter_child_axes_tag_t const& axes)
{
	PVFieldsBaseFilter::set_children_axes_tag(axes);
	_col_proto = axes.value(PVAXIS_TAG_PROTOCOL, -1);
	_col_subdomain = axes.value(PVAXIS_TAG_SUBDOMAIN, -1);
	_col_host = axes.value(PVAXIS_TAG_HOST, -1);
	_col_domain = axes.value(PVAXIS_TAG_DOMAIN, -1);
	_col_tld = axes.value(PVAXIS_TAG_TLD, -1);
	_col_port = axes.value(PVAXIS_TAG_PORT, -1);
	_col_url = axes.value(PVAXIS_TAG_URL, -1);
	_col_variable = axes.value(PVAXIS_TAG_URL_VARIABLES, -1);
	_col_fragment = axes.value(PVAXIS_TAG_URL_FRAGMENT, -1);
	_col_credentials = axes.value(PVAXIS_TAG_URL_CREDENTIALS, -1);
	PVCol nmiss = (_col_proto == -1) + (_col_subdomain == -1) + (_col_host == -1) + (_col_domain == -1) + (_col_tld == -1) + (_col_port == -1) + \
				(_col_url == -1) + (_col_variable == -1) + (_col_fragment == -1) + (_col_credentials == -1);
	_ncols = URL_NUMBER_FIELDS_CREATED-nmiss;
	if (_ncols == 0) {
		PVLOG_WARN("(PVFieldSplitterURL::set_children_axes_tag) warning: URL splitter set but no tags have been found !\n");
	}
}

static bool set_field(int pos, PVCore::PVField** fields, const uint16_t* str, furl_feature_t ff)
{
	if (pos == -1) {
		return false;
	}

	PVCore::PVField* new_f = fields[pos];
	if (furl_features_exist(ff)) {
		const uint16_t* field_str = str + ff.pos;
		new_f->set_begin((char*) field_str);
		new_f->set_end((char*) (field_str + ff.size));
		new_f->set_physical_end((char*) (field_str + ff.size));
	}
	else {
		new_f->set_begin((char*) &empty_str);
		new_f->set_end((char*) (&empty_str+1));
		new_f->set_physical_end((char*) (&empty_str+1));
	}

	return true;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterURL::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldSplitterURL::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	// furl handler
	furl_handler_t* fh = &_furl_handler.local();
	const uint16_t* str_url = (const uint16_t*) field.begin();
	if (furl_decode(fh, str_url, field.size()/sizeof(uint16_t)) != 0) {
		field.set_invalid();
		return 0;
	}

	// Add the number of final fields and save their pointers
	PVCore::PVField *pf[URL_NUMBER_FIELDS_CREATED];
	PVCore::PVField ftmp(*field.elt_parent());
	for (PVCol i = 0; i < _ncols; i++) {
		PVCore::list_fields::iterator it_new = l.insert(it_ins, ftmp);
		pf[i] = &(*it_new);
	}

	PVCore::list_fields::size_type ret = 0;
	ret += set_field(_col_proto, pf, str_url, fh->furl.features.scheme); 
	ret += set_field(_col_subdomain, pf, str_url, fh->furl.features.subdomain); 
	ret += set_field(_col_domain, pf, str_url, fh->furl.features.domain); 
	ret += set_field(_col_host, pf, str_url, fh->furl.features.host); 
	ret += set_field(_col_tld, pf, str_url, fh->furl.features.tld); 
	ret += set_field(_col_url, pf, str_url, fh->furl.features.resource_path);
	ret += set_field(_col_variable, pf, str_url, fh->furl.features.query_string); 
	ret += set_field(_col_fragment, pf, str_url, fh->furl.features.fragment); 
	ret += set_field(_col_credentials, pf, str_url, fh->furl.features.credential); 
	if (furl_features_exist(fh->furl.features.port)) {
		ret += set_field(_col_port, pf, str_url, fh->furl.features.port);
	}
	else
	if (_col_port >= 0) {
		// Guess default port from protocol
		PVCore::PVUnicodeString proto(str_url + fh->furl.features.scheme.pos, fh->furl.features.scheme.size);
		const PVCore::PVUnicodeString::utf_char* str_port; size_t size_port;
		if (proto.compareNoCase(g_str_http) == 0) {
			str_port = g_port_80;
			size_port = 2;
		}
		else
		if (proto.compareNoCase(g_str_https) == 0) {
			str_port = g_port_443;
			size_port = 3;
		}
		else
		if (proto.compareNoCase(g_str_ftp) == 0) {
			str_port = g_port_21;
			size_port = 2;
		}
		else {
			return ret;
		}

		PVCore::PVField* fport = pf[_col_port];
		fport->set_begin((char*) str_port);
		fport->set_end((char*) (str_port + size_port));
		fport->set_physical_end((char*) (str_port + size_port));
		ret++;
	}

	return ret;
}


IMPL_FILTER_NOPARAM(PVFilter::PVFieldSplitterURL)
