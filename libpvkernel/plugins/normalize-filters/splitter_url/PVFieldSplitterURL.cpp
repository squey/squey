/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Handles urls like this:
// lintranet.beijaflore.com:443
// clients1.google.com:443

#include "PVFieldSplitterURL.h"
#include <pvkernel/core/PVBufferSlice.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVAxisTagsDec.h>

#include <furl/decode.h>

static char empty_str = 0;
static constexpr const char* str_http = "http";
static constexpr const char* str_https = "https";
static constexpr const char* str_ftp = "ftp";
static constexpr const char* str_port_80 = "80";
static constexpr const char* str_port_443 = "443";
static constexpr const char* str_port_21 = "21";

#define URL_NUMBER_FIELDS_CREATED 10

/******************************************************************************
 *
 * PVFilter::PVCore::PVFieldSplitterURL::PVCore::PVFieldSplitterURL
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterURL::PVFieldSplitterURL() : PVFieldsFilter<PVFilter::one_to_many>()
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

void PVFilter::PVFieldSplitterURL::set_children_axes_tag(filter_child_axes_tag_t const& axes,
                                                         size_t fields_count)
{
	PVFieldsBaseFilter::set_children_axes_tag(axes, fields_count);
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

	_ncols = fields_count;
}

static void set_field(int pos, PVCore::PVField** fields, char* str, furl_feature_t ff)
{
	if (pos == -1) {
		return;
	}

	PVCore::PVField* new_f = fields[pos];
	if (furl_features_exist(ff)) {
		char* field_str = str + ff.pos;
		new_f->set_begin(field_str);
		new_f->set_end(field_str + ff.size);
		new_f->set_physical_end(field_str + ff.size);
	}
}

/**
 * Add port from url if available or try to guess it from protocol.
 */
static void
add_port(int pos, PVCore::PVField** fields, char* str, furl_feature_t ff, furl_feature_t ff_proto)
{
	if (pos == -1) {
		return;
	}

	PVCore::PVField* new_f = fields[pos];
	if (furl_features_exist(ff)) {
		char* field_str = str + ff.pos;
		new_f->set_begin(field_str);
		new_f->set_end(field_str + ff.size);
		new_f->set_physical_end(field_str + ff.size);
	} else {
		// Guess default port from protocol
		std::string proto(str + ff_proto.pos, ff_proto.size);
		const char* str_port;
		size_t size_port;
		if (proto == str_http) {
			str_port = str_port_80;
			size_port = 2;
		} else if (proto == str_https) {
			str_port = str_port_443;
			size_port = 3;
		} else if (proto == str_ftp) {
			str_port = str_port_21;
			size_port = 2;
		} else {
			return;
		}

		new_f->set_begin((char*)str_port);
		new_f->set_end((char*)(str_port + size_port));
		new_f->set_physical_end((char*)(str_port + size_port));
	}
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterURL::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldSplitterURL::one_to_many(
    PVCore::list_fields& l, PVCore::list_fields::iterator it_ins, PVCore::PVField& field)
{
	// furl handler
	furl_handler_t* fh = &_furl_handler.local();
	char* str_url = field.begin();
	if (furl_decode(fh, str_url, field.size()) != 0) {
		field.set_invalid();
		return 0;
	}

	// Add the number of final fields and save their pointers
	PVCore::PVField* pf[URL_NUMBER_FIELDS_CREATED];
	const PVCore::PVField null_field(*field.elt_parent(), &empty_str, &empty_str);
	for (PVCol i = 0; i < _ncols; i++) {
		PVCore::list_fields::iterator it_new = l.insert(it_ins, null_field);
		pf[i] = &(*it_new);
	}

	set_field(_col_proto, pf, str_url, fh->furl.features.scheme);
	set_field(_col_subdomain, pf, str_url, fh->furl.features.subdomain);
	set_field(_col_domain, pf, str_url, fh->furl.features.domain);
	set_field(_col_host, pf, str_url, fh->furl.features.host);
	set_field(_col_tld, pf, str_url, fh->furl.features.tld);
	set_field(_col_url, pf, str_url, fh->furl.features.resource_path);
	set_field(_col_variable, pf, str_url, fh->furl.features.query_string);
	set_field(_col_fragment, pf, str_url, fh->furl.features.fragment);
	set_field(_col_credentials, pf, str_url, fh->furl.features.credential);
	add_port(_col_port, pf, str_url, fh->furl.features.port, fh->furl.features.scheme);

	return _ncols;
}
