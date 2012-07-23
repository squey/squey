/**
 * \file type-discovery.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#if 0

#include <picviz/type-discovery.h>

#include <stdio.h>
#include <string.h>
#include <pcre.h>

#define IPV4_PCRE "\\d+.\\d+.\\d+.\\d+"
#define INTEGER_PCRE "[0-9]+"
#define FLOAT_PCRE "[0-9\.]+"
#define TIME24H_PCRE "\\d+:\\d+:\\d+"

#define PCRE_OVECTOR_SIZE 30

static int picviz_type_discovery_pcre_check(char *pcrepattern, char *input)
{
	pcre *re;
	const char *error;
	int erroffset;
	int ovector[PCRE_OVECTOR_SIZE];

	const char **stringlist;
	int stringcount;

	int retval;

	re = pcre_compile(pcrepattern, 0, &error, &erroffset, NULL);
	if ( ! re ) {
		fprintf(stderr, "Unable to compile regex[offset:%d]: %s.\n", erroffset, error);
		return 0;
	}
	stringcount = pcre_exec(re, NULL, input, strlen(input), 0, 0, ovector, PCRE_OVECTOR_SIZE);
        if ( stringcount >= 0 ) {
		retval = pcre_get_substring_list((const char *)input, ovector, stringcount, &stringlist);

		/* And we will match 11:30:12 -> 11 */
		if (strlen(stringlist[0]) != strlen(input)) {
			/* So we don't want 11:30:12 to appear like an integer */
			pcre_free(re);
			return 0;
		}

		pcre_free(re);
		return 1;
	}

        pcre_free(re);
	return 0;
}

static int picviz_type_discovery_is_integer(char *input)
{
	return picviz_type_discovery_pcre_check(INTEGER_PCRE, input);
}

static int picviz_type_discovery_is_time24h(char *input)
{
	return picviz_type_discovery_pcre_check(TIME24H_PCRE, input);
}

static int picviz_type_discovery_is_ipv4(char *input)
{
	return picviz_type_discovery_pcre_check(IPV4_PCRE, input);
}

static int picviz_type_discovery_is_float(char *input)
{
	return picviz_type_discovery_pcre_check(FLOAT_PCRE, input);
}

char *picviz_type_discovery_do_one(char *input)
{
	if (picviz_type_discovery_is_integer(input)) {
		/* printf("input '%s' is an integer\n", input); */
		return "integer";
	}
	if (picviz_type_discovery_is_time24h(input)) {
		/* printf("input '%s' is a time 24h\n", input); */
		return "time";
	}
	if (picviz_type_discovery_is_ipv4(input)) {
		/* printf("input '%s' is a ipv4\n", input); */
		return "ipv4";
	}
	if (picviz_type_discovery_is_float(input)) {
		/* printf("input '%s' is a float\n", input); */
		return "float";
	}

	return "string";
}

#endif

#ifdef _UNIT_TEST_
int main(void)
{
	char *type;

	type = picviz_type_discovery_do_one("192.168.12.3");
	printf("Discovered type:%s\n", type);

	return 0;
}
#endif
