/**
 * \file PVFieldMacAddress.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVFieldMacAddress.h"

/**
 * RH: the created fields must *not* be terminated by a '\0', otherwise, it
 * moves the fields in an other PVElement... 2 days to understand that...
 */
const char* PVFilter::PVFieldMacAddress::uppercased_str = "uppercase";

static char char_low[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };
static char char_upper[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };

/******************************************************************************
 * get_hex_char
 *****************************************************************************/

static inline int get_hex_char(uint16_t value)
{
	if ((value >= '0') && (value <= '9')) {
		return value - '0';
	} else if ((value >= 'a') && (value <= 'f')) {
		return (value - 'a') + 10;
	} else if ((value >= 'A') && (value <= 'F')) {
		return (value - 'A') + 10;
	} else {
		return -1;
	}
}

/******************************************************************************
 * hmac_scan_6b
 *****************************************************************************/

#define GET(RES, INDEX) \
	if ((RES = get_hex_char(text[(INDEX)])) < 0) { \
		return 0; \
	}

// extract hmac info from form aa:bb:cc:dd:ee:ff
static int hmac_scan_6b(const uint16_t* text,
                        int* v0, int* v1, int* v2,
                        int* d0, int* d1, int* d2)
{
#ifdef USE_UTF_8
	// use the discard directive "%*" to ignore the separators
	return sscanf(text, "%02x%*c%02x%*c%02x%*c%02x%*c%02x%*c%02x",
	              v0, v1, v, d0, d1, d1);
#else
	int a, b;

	GET(a, 0);
	GET(b, 1);
	*v0 = (a << 4) + b;

	GET(a, 3);
	GET(b, 4);
	*v1 = (a << 4) + b;

	GET(a, 6);
	GET(b, 7);
	*v2 = (a << 4) + b;

	GET(a, 9);
	GET(b, 10);
	*d0 = (a << 4) + b;

	GET(a, 12);
	GET(b, 13);
	*d1 = (a << 4) + b;

	GET(a, 15);
	GET(b, 16);
	*d2 = (a << 4) + b;

	return 6;
#endif
}

/******************************************************************************
 * hmac_scan_3u
 *****************************************************************************/

// extract hmac info from form aaaa.bbbb.cccc
static int hmac_scan_3u(const uint16_t* text,
                        int* v0, int* v1, int* v2,
                        int* d0, int* d1, int* d2)
{
#ifdef USE_UTF_8
	return sscanf(text, "%02x%02x.%02x%02x.%02x%02x",
	              v0, v1, v2,
	              d0, d1, d2);
#else
	int a, b;

	GET(a, 0);
	GET(b, 1);
	*v0 = (a << 4) + b;

	GET(a, 2);
	GET(b, 3);
	*v1 = (a << 4) + b;

	GET(a, 5);
	GET(b, 6);
	*v2 = (a << 4) + b;

	GET(a, 7);
	GET(b, 8);
	*d0 = (a << 4) + b;

	GET(a, 10);
	GET(b, 11);
	*d1 = (a << 4) + b;

	GET(a, 12);
	GET(b, 13);
	*d2 = (a << 4) + b;

	return 6;
#endif
}

#undef GET

/******************************************************************************
 * hmac_print_3b
 *****************************************************************************/

#define SET_SEP(INDEX, VALUE)                                           \
	text[(2*(INDEX))+0] = (VALUE); \
	text[(2*(INDEX))+1] = 0; \

#define SET_VAL(INDEX, VALUE)                                           \
	text[(2*(INDEX))+0] = array[((VALUE) >> 4)]; \
	text[(2*(INDEX))+1] = 0; \
	text[(2*(INDEX))+2] = array[((VALUE) & 0x0000000f)]; \
	text[(2*(INDEX))+3] = 0;

static void hmac_print_3b(char* text, char* array,
                          int v0, int v1, int v2)
{
#ifdef USE_UTF_8
	sprintf(text, "%02x:%02x:%02x", v0, v1, v2);
#else

	SET_VAL(0, v0);
	SET_SEP(2, ':');
	SET_VAL(3, v1);
	SET_SEP(5, ':');
	SET_VAL(6, v2);
#endif
}

#undef SET_SEP
#undef SET_VAL

/******************************************************************************
 * hmac_extract
 *****************************************************************************/

static int hmac_extract(PVCore::PVField &field,
                        int *ven_0, int *ven_1, int *ven_2,
                        int *dev_0, int *dev_1, int *dev_2)
{
	/**
	 * the text is in UTF-16 and field.begin() return a char*...
	 *
	 * As in C/C++, a litteral char ('x') is considered as an int, it can
	 * be implicitly compared with a uint16_t
	 */
#ifdef USE_UTF_8
	const char *text = field.begin();
	int len = field.size();
#else
	const uint16_t *text = (uint16_t*)field.begin();
	int len = field.size() / 2;
#endif

	/**
	 * according to http://en.wikipedia.org/wiki/MAC_address, there are 3
	 * different formats for MAC address:
	 * - 01:23:45:67:89:ab
	 * - 01-23-45-67-89-ab
	 * - 0123.4567.89ab
	 */

	// ordering from the more regular case to the less one
	if (len == 17) {
		char sep = text[2];
		if ((sep != '-') && (sep != ':')) {
			// malformed: first separator is invalid (must match '[-:]')
			return 0;
		}

		for (int i = 5; i <=14; i += 3) {
			if (text[i] != sep) {
				/* malformed: next separator has changed (must
				 * previously found separator
				 */
				return 0;
			}
		}

		if (hmac_scan_6b(text,
		                 ven_0, ven_1, ven_2,
		                 dev_0, dev_1, dev_2) != 6) {
			// malformed: must be 6 hexadecimal bytes
			return 0;
		}
	} else if (len != 14) {
		// invalid length
		return 0;
	} else {
		if ((text[4] != '.') || (text[9] != '.')) {
			// malformed: not '.'
			return 0;
		} else if (hmac_scan_3u(text,
		                        ven_0, ven_1, ven_2,
		                        dev_0, dev_1, dev_2) != 6) {
			// malformed: must be 6 hexadecimal bytes
			return 0;
		}
	}

	return 2;
}

/******************************************************************************
 * PVFilter::PVFieldMacAddress::PVFieldMacAddress
 *****************************************************************************/

PVFilter::PVFieldMacAddress::PVFieldMacAddress(PVCore::PVArgumentList const& args) :
	PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER(PVFilter::PVFieldMacAddress, args);
}

/******************************************************************************
 * PVFilter::PVFieldMacAddress::set_args
 *****************************************************************************/

void PVFilter::PVFieldMacAddress::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_uppercased = args[uppercased_str].toBool();
}

/******************************************************************************
 * DEFAULT_ARGS_FILTER
 *****************************************************************************/

DEFAULT_ARGS_FILTER(PVFilter::PVFieldMacAddress)
{
	PVCore::PVArgumentList args;
	args[uppercased_str] = false;
	return args;
}

/******************************************************************************
 * PVFilter::PVFieldMacAddress::one_to_many
 *****************************************************************************/

PVCore::list_fields::size_type
PVFilter::PVFieldMacAddress::one_to_many(PVCore::list_fields &l,
                                         PVCore::list_fields::iterator it_ins,
                                         PVCore::PVField &field)
{
	int ven_0;
	int ven_1;
	int ven_2;

	int dev_0;
	int dev_1;
	int dev_2;

	if (hmac_extract(field,
	                 &ven_0, &ven_1, &ven_2,
	                 &dev_0, &dev_1, &dev_2) == 0) {
		field.set_invalid();
		return 0;
	}

	PVCore::PVField &ven_f(*l.insert(it_ins, field));
	PVCore::PVField &dev_f(*l.insert(it_ins, field));

#ifdef USE_UTF_8
	const int hmac_block_size = 8; // 9 * sizeof(char)
#else
	const int hmac_block_size = 16; // 9 * sizeof(uint16_t)
#endif

	ven_f.allocate_new(hmac_block_size);
	if (_uppercased) {
		hmac_print_3b(ven_f.begin(), char_upper, ven_0, ven_1, ven_2);
	} else {
		hmac_print_3b(ven_f.begin(), char_low, ven_0, ven_1, ven_2);
	}
	ven_f.set_end(ven_f.begin() + hmac_block_size);

	dev_f.allocate_new(hmac_block_size);
	if (_uppercased) {
		hmac_print_3b(dev_f.begin(), char_upper, dev_0, dev_1, dev_2);
	} else {
		hmac_print_3b(dev_f.begin(), char_low, dev_0, dev_1, dev_2);
	}
	dev_f.set_end(dev_f.begin() + hmac_block_size);

	return 2;
}

/******************************************************************************
 * IMPL_FILTER
 *****************************************************************************/

IMPL_FILTER(PVFilter::PVFieldMacAddress)
