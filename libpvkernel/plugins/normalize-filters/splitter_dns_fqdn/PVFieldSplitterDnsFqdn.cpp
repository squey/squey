/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

/**
 * about this splitter
 *
 * It permit to split a FQDN in a DNS context into:
 * - TLD1: 'com' in 'www.en.example.com'
 * - TLD2: 'example.com' in 'www.en.example.com'
 * - TLD3: 'en.example.com' in 'www.en.example.com'
 * - SUBD1: 'www.en.example' in 'www.en.example.com'
 * - SUBD2: 'www.en' in 'www.en.example.com'
 * - SUBD3: 'www' in 'www.en.example.com'
 *
 * In case of PTR DNS query, the third level domain is an IP address (the
 * SLD+TLD is also "in-addr.arpa" for IPv4 and "ip6.arpa" for IPv6); the
 * fields behavior is also:
 * - TLD3 remains empty
 * - IP addresses appear in the SUBD* fields
 *
 * When SUBD* contain IPs, they can be reversed to look in the "right order".
 *
 * Note: A FQDN may start with a dot ('.') but it is insightful. As it can
 * have a meaning for the user, it is not removed.
 *
 * Note 2: A FQDN may end with a dot ('.') but it is insightful. While it has
 * a meaning while processing a hostname (to expand it with the domain name
 * or not), it has none in any other cases. It is also removed from the
 * splitting proccess (and result).
 *
 * Note 3: TLDs like "co.uk" or "notaires.fr" are not handled by this
 * splitter.
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

	_n         = args.at(N).toInt();
	_tld1      = args.at(TLD1).toBool();
	_tld2      = args.at(TLD2).toBool();
	_tld3      = args.at(TLD3).toBool();
	_subd1     = args.at(SUBD1).toBool();
	_subd2     = args.at(SUBD2).toBool();
	_subd3     = args.at(SUBD3).toBool();
	_subd1_rev = args.at(SUBD1_REV).toBool();
	_subd2_rev = args.at(SUBD2_REV).toBool();
	_subd3_rev = args.at(SUBD3_REV).toBool();

	_need_inv = (_subd1 && _subd1_rev) || (_subd2 && _subd2_rev) || (_subd3 && _subd3_rev);
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

/**
 * search, in the reverse order, for the start and the length of a
 * domain in a FQDN. It returns true only if the string seems to have a
 * lower level domain.
 */
static inline bool str_rscan(char *str, int &pos, int &start, int& len)
{
	len = 0;

	while ((pos >= 0) && (str[pos] != '.')) {
		--pos;
		++len;
	}

	if (pos < 0) {
		++pos;
		start = 0;
		return false;
	} else {
		assert(str[pos] == '.');
		start = pos + 1;
		pos -= 1;
		return true; // only if (pos >= 0) && (str[pos] == '.')
	}
}

static inline void check_arpa_ip(char *str,
                                 int tld1_pos, int tld1_len,
                                 int tld2_pos, int tld2_len,
                                 bool &is_ipv4, bool &is_ipv6)
{
	static char ARPA[] = "arpa";
	static char ARPA_INADDR[] = "in-addr";
	static char ARPA_IP6[] = "ip6";

	if ((tld1_len != 4) || (memcmp(str + tld1_pos, ARPA, 4) != 0)) {
		is_ipv4 = false;
		is_ipv6 = false;
	} else if ((tld2_len == 7) && (memcmp(str + tld2_pos, ARPA_INADDR, 7) == 0)) {
		is_ipv4 = true;
		is_ipv6 = false;
	} else if ((tld2_len == 3) && (memcmp(str + tld2_pos, ARPA_IP6, 3) == 0)) {
		is_ipv4 = false;
		is_ipv6 = true;
	} else {
		is_ipv4 = false;
		is_ipv6 = false;
	}
}

static void fill_field(PVCore::PVField &field, char* str, int len)
{
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

	char*str = field.begin();

	int str_len = field.size();

	int pos = str_len - 1;

	if (str[pos] == '.') {
		/* FQDN may finish with a dot to indicate that no more
		 * expansion can be done; not removing it (because it is
		 * useless) could be messy...
		 */
		--pos;
	}

	int tld1_pos = 0, tld2_pos = 0, tld3_pos = 0;
	int tld1_len = 0, tld2_len = 0, tld3_len = 0;

	bool has_tld2 = false;
	bool has_tld3 = false;

	bool is_ip = false;
	bool is_ipv4 = false;
	bool is_ipv6 = false;

	if (_tld1 || _tld2 || _tld3 || _subd1 || _subd2 || _subd3) {
		has_tld2 = str_rscan(str, pos, tld1_pos, tld1_len);
	}
	if (_tld2 || _tld3 || _subd1 || _subd2 || _subd3) {
		has_tld3 = str_rscan(str, pos, tld2_pos, tld2_len);
	}

	if (_tld3 || _subd1 || _subd2 || _subd3) {
		// knowing that a 4th LD exists is useless
		str_rscan(str, pos, tld3_pos, tld3_len);
	}

	char rev_str[64];
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

		if (is_ip && _need_inv) {
			/* compute a in-order IP only if required
			 */
			if (is_ipv4) {
				/* invert IPv4's octets (from end to start):
				 * find the first octet (the last in str),
				 * copy it, find the second, copy it...
				 *
				 * how boring O:-)
				 */
				pos = tld2_pos - 2;
				int len = 0;
				int rpos;
				int wpos = 0;
				// 1st octet
				str_rscan(str, pos, rpos, len);
				memcpy(rev_str + wpos, str + rpos, len);
				wpos += len;
				rev_str[wpos] = '.';
				wpos++;

				// 2nd octet
				len = 0;
				str_rscan(str, pos, rpos, len);
				memcpy(rev_str + wpos, str + rpos, len);
				wpos += len;
				rev_str[wpos] = '.';
				wpos++;

				// 3rd octet
				len = 0;
				str_rscan(str, pos, rpos, len);
				memcpy(rev_str + wpos, str + rpos, len);
				wpos += len;
				rev_str[wpos] = '.';
				wpos++;

				// 4th octet
				len = 0;
				str_rscan(str, pos, rpos, len);
				memcpy(rev_str + wpos, str + rpos, len);

				rev_len = wpos + len;
			} else {
				/* more simple for IPv6: each octet is
				 * represented as "X.Y" where X and Y are the
				 * 2 quartets (in hexadecimal) composing the
				 * octet and each octet's representation is
				 * separated by a '.' => 32 quartets and 31
				 * '.'.
				 * trivial to invert :-]
				 */
				rev_len = 63;
				std::reverse_copy(str, str + rev_len, rev_str);
			}
		}
	}

	int len = tld1_len;

	if (_tld1) {
		PVCore::PVField &f(*l.insert(it_ins, field));
		f.set_begin(str + tld1_pos);
		f.set_end(str + tld1_pos + len);
		++ret;
	}

	len += (has_tld2 == false)?0:(tld2_len + 1);

	if (_tld2) {
		PVCore::PVField &f(*l.insert(it_ins, field));
		// set tld1 and tld2
		f.set_begin(str + tld2_pos);
		f.set_end(str + tld2_pos + len);
		++ret;
	}

	len += (has_tld3 == false)?0:(tld3_len + 1);

	if (_tld3) {
		PVCore::PVField &f(*l.insert(it_ins, field));
		if (is_ip) {
			// Empty field.
			f.set_end(f.begin());
		} else {
			fill_field(f, str + tld3_pos, len);
		}
		++ret;
	}

	if (_subd1) {
		PVCore::PVField &f(*l.insert(it_ins, field));
		if (not is_ip and tld1_pos != 0) {
			f.set_end(f.begin() + tld1_pos - 1);
		} else if (tld2_pos != 0) {
			if (_subd1_rev) {
				fill_field(f, rev_str, rev_len);
			} else {
				f.set_end(f.begin() + tld2_pos - 1);
			}
		} else {
			// Empty field.
			f.set_end(f.begin());
		}
		++ret;
	}

	if (_subd2) {
		PVCore::PVField &f(*l.insert(it_ins, field));
		if (not is_ip) {
			if (tld2_pos != 0) {
				// there is a sub-domain under the SLD
				f.set_end(f.begin() + tld2_pos - 1);
			} else {
				// Empty field.
				f.set_end(f.begin());
			}
		} else if (tld2_pos != 0) {
			if (_subd2_rev) {
				fill_field(f, rev_str, rev_len);
			} else {
				f.set_end(f.begin() + tld2_pos - 1);
			}
		} else {
			// Empty field.
			f.set_end(f.begin());
		}
		++ret;
	}

	if (_subd3) {
		PVCore::PVField &f(*l.insert(it_ins, field));
		if (!is_ip) {
			if (tld3_pos != 0) {
				// there is a sub-domain under the 3rd LD
				f.set_end(f.begin() + tld3_pos - 1);
			} else {
				// Empty field.
				f.set_end(f.begin());
			}
		} else if (tld2_pos != 0) {
			if (_subd3_rev) {
				fill_field(f, rev_str, rev_len);
			} else {
				f.set_end(f.begin() + tld2_pos - 1);
			}
		} else {
			// Empty field.
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
