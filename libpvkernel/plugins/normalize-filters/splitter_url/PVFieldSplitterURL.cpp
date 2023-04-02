//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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
	set_number_expected_fields(URL_NUMBER_FIELDS_CREATED);
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
	for (auto & i : pf) {
		PVCore::list_fields::iterator it_new = l.insert(it_ins, null_field);
		i = &(*it_new);
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
