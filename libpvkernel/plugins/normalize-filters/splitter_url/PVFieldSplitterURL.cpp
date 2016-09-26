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

#include <furl/decode.h>

static char empty_str = 0;
static constexpr const char* str_http = "http";
static constexpr const char* str_https = "https";
static constexpr const char* str_ftp = "ftp";
static constexpr const char* str_port_80 = "80";
static constexpr const char* str_port_443 = "443";
static constexpr const char* str_port_21 = "21";

static constexpr size_t URL_NUMBER_FIELDS_CREATED = 10;

/******************************************************************************
 *
 * PVFilter::PVCore::PVFieldSplitterURL::PVCore::PVFieldSplitterURL
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterURL::PVFieldSplitterURL() : PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER_NOPARAM(PVFilter::PVFieldSplitterURL);
}

static void set_field(int pos, PVCore::PVField** fields, char* str, furl_feature_t ff)
{
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
	for (size_t i = 0; i < URL_NUMBER_FIELDS_CREATED; i++) {
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

	return URL_NUMBER_FIELDS_CREATED;
}
