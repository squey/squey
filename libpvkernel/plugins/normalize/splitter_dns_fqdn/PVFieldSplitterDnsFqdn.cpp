/**
 * \file PVFieldSplitterDnsFqdn.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

/**
 * about this splitter
 *
 * It permit to split a FQDN in a DNS context into:
 * - TLD1: the first top level domain
 * - TLD2: the two first top level domains
 * - TLD3: the three first top level domains
 * - SUBD1: all but the first top level domain
 * - SUBD2: all but the two first top level domains
 * - SUBD3: all but the three first top level domains
 *
 * In case of PTR DNS query, the third level domain is an IP address (the
 * SLD+TLD is also "in-addr.arpa" for IPv4 and "ip6.arpa" for IPv6); the
 * fields behavior is also:
 * - TLD3 remains empty
 * - IP addresses appear in the SUBD* fields
 *
 * When SUBD* contain IPs, they can be reversed to look in the "right order".
 *
 * Note: the involved TLD (top level domain) concept has be understand in its
 * original form: the token after the last dot ('.'). More recent TLD concept
 * like "co.uk" or "notaires.fr" are not handled as special cases.
 */
#include "PVFieldSplitterDnsFqdn.h"

const char* PVFilter::PVFieldSplitterDnsFqdn::N         = "n";
const char* PVFilter::PVFieldSplitterDnsFqdn::TLD1      = "tld1";
const char* PVFilter::PVFieldSplitterDnsFqdn::TLD2      = "tld2";
const char* PVFilter::PVFieldSplitterDnsFqdn::TLD3      = "tld3";
const char* PVFilter::PVFieldSplitterDnsFqdn::SUBD1     = "subd1";
const char* PVFilter::PVFieldSplitterDnsFqdn::SUBD2     = "subd2";
const char* PVFilter::PVFieldSplitterDnsFqdn::SUBD3     = "subd3";
const char* PVFilter::PVFieldSplitterDnsFqdn::SUBD1_REV = "subd1_rev";
const char* PVFilter::PVFieldSplitterDnsFqdn::SUBD2_REV = "subd2_rev";
const char* PVFilter::PVFieldSplitterDnsFqdn::SUBD3_REV = "subd3_rev";

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdn::PVFieldSplitterDnsFqdn
 *****************************************************************************/

PVFilter::PVFieldSplitterDnsFqdn::PVFieldSplitterDnsFqdn(PVCore::PVArgumentList const& args) :
	PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER(PVFilter::PVFieldSplitterDnsFqdn, args);
}

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdn::set_args
 *****************************************************************************/

void PVFilter::PVFieldSplitterDnsFqdn::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);

	_n         = args[N].toInt();
	_tld1      = args[TLD1].toBool();
	_tld2      = args[TLD2].toBool();
	_tld3      = args[TLD3].toBool();
	_subd1     = args[SUBD1].toBool();
	_subd2     = args[SUBD2].toBool();
	_subd3     = args[SUBD3].toBool();
	_subd1_rev = args[SUBD1_REV].toBool();
	_subd2_rev = args[SUBD2_REV].toBool();
	_subd3_rev = args[SUBD3_REV].toBool();

	_need_rev = (_subd1 && _subd1_rev) || (_subd2 && _subd2_rev) || (_subd3 && _subd3_rev);
}

/******************************************************************************
 * DEFAULT_ARGS_FILTER
 *****************************************************************************/

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterDnsFqdn)
{
	PVCore::PVArgumentList args;

	args[N]         = 2;
	args[TLD1]      = true;
	args[TLD2]      = false;
	args[TLD3]      = false;
	args[SUBD1]     = true;
	args[SUBD2]     = false;
	args[SUBD3]     = false;
	args[SUBD1_REV] = true;
	args[SUBD2_REV] = true;
	args[SUBD3_REV] = true;

	return args;
}

static inline int str_rscan(uint16_t *str, int &pos, int &rpos)
{
	int len = 0;

	if (pos == 0) {
		rpos = 0;
		return 0;
	}

	while ((pos >= 0) && (str[pos] != '.')) {
		--pos;
		++len;
	}

	if (pos < 0) {
		++pos;
		rpos = 0;
	} else if (str[pos] == '.') {
		rpos = pos + 1;
		pos -= 1;
	} else {
		--len;
	}

	return len;
}

static inline void check_arpa_ip(uint16_t *str,
                                 int tld1_pos, int tld1_len,
                                 int tld2_pos, int tld2_len,
                                 bool &is_ipv4, bool &is_ipv6)
{
	// UTF16 LE strings definition
	static uint16_t ARPA[]        = { 'a', 'r', 'p', 'a' };
	static uint16_t ARPA_INADDR[] = { 'i', 'n', '-', 'a', 'd', 'd', 'r' };
	static uint16_t ARPA_IP6[]    = { 'i', 'p', '6' };

	if ((tld1_len != 4) || (memcmp(str + tld1_pos, ARPA, 8) != 0)) {
		is_ipv4 = false;
		is_ipv6 = false;
	} else if ((tld2_len == 7) && (memcmp(str + tld2_pos, ARPA_INADDR, 14) == 0)) {
		is_ipv4 = true;
		is_ipv6 = false;
	} else if ((tld2_len == 3) && (memcmp(str + tld2_pos, ARPA_IP6, 6) == 0)) {
		is_ipv4 = false;
		is_ipv6 = true;
	} else {
		is_ipv4 = false;
		is_ipv6 = false;
	}
}

static void fill_field(PVCore::PVField &field, uint16_t* str, int len)
{
	len *= 2;

	field.allocate_new(len);
	if (len) {
		memcpy(field.begin(), str, len);
	}
	field.set_end(field.begin() + len);
}

/******************************************************************************
 * PVFilter::PVFieldSplitterDnsFqdn::one_to_many
 *****************************************************************************/

PVCore::list_fields::size_type
PVFilter::PVFieldSplitterDnsFqdn::one_to_many(PVCore::list_fields &l,
                                              PVCore::list_fields::iterator it_ins,
                                              PVCore::PVField &field)
{
	PVCore::list_fields::size_type ret = 0;

	uint16_t *str = (uint16_t*)field.begin();

	int str_len = field.size() / 2;

	int pos = str_len - 1;

	int tld1_pos = 0, tld2_pos = 0, tld3_pos = 0;
	int tld1_len = 0, tld2_len = 0, tld3_len = 0;

	bool is_ip = false;
	bool is_ipv4 = false;
	bool is_ipv6 = false;

	if (_tld1 || _tld2 || _tld3 || _subd1 || _subd2 || _subd3) {
		tld1_len = str_rscan(str, pos, tld1_pos);
	}
	if (_tld2 || _tld3 || _subd1 || _subd2 || _subd3) {
		tld2_len = str_rscan(str, pos, tld2_pos);
	}

	if (_tld3 || _subd1 || _subd2 || _subd3) {
		tld3_len = str_rscan(str, pos, tld3_pos);
	}

	uint16_t rev_str[64];
	int rev_len = 0;

	if (tld3_len != 0) {
		/**
		 * an IP can only appears as 3rd LD when the SLD+TLD is:
		 * - in-addr.arpa (for IPv4)
		 * - ip6.arpa (for IPv6)
		 * otherwise, it is a sub-domain
		 */
		check_arpa_ip(str, tld1_pos, tld1_len, tld2_pos, tld2_len, is_ipv4, is_ipv6);
		is_ip = is_ipv4 || is_ipv6;

		if (is_ip && _need_rev) {
			/* compute a in-order IP only if required
			 */
			if (is_ipv4) {
				// TODO : write IPv4 reversing
				rev_str[0] = 't';
				rev_str[1] = 'o';
				rev_str[2] = 'd';
				rev_str[3] = 'o';
				rev_len = 4;

				pos = tld2_pos - 2;
				int len, rpos;
				int wpos = 0;
				// 4th octet
				len = str_rscan(str, pos, rpos);
				for(int i = 0; i < len; ++i) { rev_str[wpos + i] = str[rpos + i]; }
				wpos += len;
				rev_str[wpos] = '.';
				wpos += 1;

				// 3rd octet
				len = str_rscan(str, pos, rpos);
				for(int i = 0; i < len; ++i) { rev_str[wpos + i] = str[rpos + i]; }
				wpos += len;
				rev_str[wpos] = '.';
				wpos += 1;

				// 2nd octet
				len = str_rscan(str, pos, rpos);
				for(int i = 0; i < len; ++i) { rev_str[wpos + i] = str[rpos + i]; }
				wpos += len;
				rev_str[wpos] = '.';
				wpos += 1;

				// 1st octet
				len = str_rscan(str, pos, rpos);
				for(int i = 0; i < len; ++i) { rev_str[wpos + i] = str[rpos + i]; }

				rev_len = wpos + len;
			} else { // is_ipv6 == true
				for(int i = 0; i < 63; ++i) {
						rev_str[i] = str[62 - i];
				}
				rev_len = 63;
			}
		}
	}

	int len = tld1_len;

	if (_tld1) {
		PVCore::PVField &f(*l.insert(it_ins, field));
		if (!is_ip) {
			fill_field(f, str + tld1_pos, len);
		} else {
			f.allocate_new(0);
			f.set_end(f.begin());
		}
		++ret;
	}

	len += (tld2_len == 0)?0:(tld2_len + 1);

	if (_tld2) {
		PVCore::PVField &f(*l.insert(it_ins, field));
		if (!is_ip) {
			fill_field(f, str + tld2_pos, len);
		} else {
			f.allocate_new(0);
			f.set_end(f.begin());
		}
		++ret;
	}

	len += (tld3_len == 0)?0:(tld3_len + 1);

	if (_tld3) {
		PVCore::PVField &f(*l.insert(it_ins, field));
		if (!is_ip) {
			fill_field(f, str + tld3_pos, len);
		} else {
			f.allocate_new(0);
			f.set_end(f.begin());
		}
		++ret;
	}

	if (_subd1) {
		PVCore::PVField &f(*l.insert(it_ins, field));
		if (!is_ip) {
			if (tld1_pos != 0) {
				fill_field(f, str, tld1_pos - 1);
			} else {
				f.allocate_new(0);
				f.set_end(f.begin());
			}
		} else  if (tld2_pos != 0) {
			if (_subd1_rev) { // an in-order IP
				fill_field(f, rev_str, rev_len);
			} else { // a reversed IP
				fill_field(f, str, tld2_pos - 1);
			}
		} else {
			f.allocate_new(0);
			f.set_end(f.begin());
		}
		++ret;
	}

	if (_subd2) {
		PVCore::PVField &f(*l.insert(it_ins, field));
		if (!is_ip) {
			if (tld2_pos != 0) {
				// there is a sub-domain under the SLD
				fill_field(f, str, tld2_pos - 1);
			} else {
				f.allocate_new(0);
				f.set_end(f.begin());
			}
		} else if (tld2_pos != 0) {
			if (_subd2_rev) { // an in-order IP
				fill_field(f, rev_str, rev_len);
			} else { // a reversed IP
				fill_field(f, str, tld2_pos - 1);
			}
		} else {
			f.allocate_new(0);
			f.set_end(f.begin());
		}
		++ret;
	}

	if (_subd3) {
		PVCore::PVField &f(*l.insert(it_ins, field));
		if (!is_ip) {
			if (tld3_pos != 0) {
				// there is a sub-domain under the 3rd LD
				fill_field(f, str, tld3_pos - 1);
			} else {
				f.allocate_new(0);
				f.set_end(f.begin());
			}
		} else if (tld2_pos != 0) {
			if (_subd3_rev) { // an in-order IP
				fill_field(f, rev_str, rev_len);
			} else { // a reversed IP
				fill_field(f, str, tld2_pos - 1);
			}
		} else {
			f.allocate_new(0);
			f.set_end(f.begin());
		}
		++ret;
	}

	return ret;
}

/******************************************************************************
 * IMPL_FILTER
 *****************************************************************************/

IMPL_FILTER(PVFilter::PVFieldSplitterDnsFqdn)
