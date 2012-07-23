//! \file PVCore::PVFieldSplitterURL.cpp
//! $Id: PVFieldSplitterURL.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2012
//! Copyright (C) Philippe Saadé 2011-2012
//! Copyright (C) Picviz Labs 2011-2012

// Handles urls like this:
// lintranet.beijaflore.com:443
// clients1.google.com:443

#include "PVFieldSplitterURL.h"
#include <pvkernel/core/PVBufferSlice.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVAxisTagsDec.h>

#include <furl/decode.h>

#include <QUrl>

const uint16_t empty_str = 0;

#define URL_NUMBER_FIELDS_CREATED 8

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
	_ncols = 8;
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
	PVCol nmiss = (_col_proto == -1) + (_col_subdomain == -1) + (_col_host == -1) + (_col_domain == -1) + (_col_tld == -1) + (_col_port == -1) + \
				(_col_url == -1) + (_col_variable == -1);
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
	ret += set_field(_col_port, pf, str_url, fh->furl.features.port);

	return ret;
}


IMPL_FILTER_NOPARAM(PVFilter::PVFieldSplitterURL)
