#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifdef WIN32
#include <float.h> // for _logb()
#endif

#include <apr_hash.h>
#include <apr_pools.h>

#include <picviz/general.h>
#include <picviz/function.h>
#include <picviz/mapping.h>

#include <picviz/PVMapping.h>

static float _enum_position_factorize(int enumber)
{
	float res = 0;
#ifdef WIN32
	int N = _logb(enumber);
#else
	int N = ilogb(enumber);
#endif
	int i;
	int x = enumber;

	if ( ! enumber) return -1;
	
	for (i = 0; i != N+1; i++) {
		if (x%2 == 0) {
			res = 2 * res;
		} else {
			res = 1+2*res;
		}
		x = x >> 1;
	}
	
	res = res / pow(2, N+1);
	
	return res;
}

/* Add is_last to deallocate the hash! */
LibCPPExport float picviz_mapping_exec(const Picviz::PVMapping_p mapping, PVCol index, QString &value, void *userdata, bool is_first)
{
struct enum_hash_t {
	apr_hash_t *values;
	int poscount;
};
        struct enum_hash_t *enum_hash;

	float retval;
	apr_uint64_t position = 0;

	size_t val_len;
	char store_c_10;

	picviz_mapping_t *mapping_o = (picviz_mapping_t *)mapping;

	val_len = value.size();
	if (val_len > 10) {
		store_c_10 = value[10];
		value[10] = '\0';
	}

	if (is_first) {
		enum_hash = (struct enum_hash_t *)malloc(sizeof(struct enum_hash_t));
		enum_hash->values = (apr_hash_t *)apr_hash_make(mapping_o->pool);
		enum_hash->poscount = 1;
		apr_hash_set(enum_hash->values, value, APR_HASH_KEY_STRING, (const void *)enum_hash->poscount);
		PICVIZ_USERDATA(userdata, struct enum_hash_t *) = enum_hash;
		retval = _enum_position_factorize(enum_hash->poscount);
	} else {
		enum_hash = PICVIZ_USERDATA(userdata, struct enum_hash_t *);
		position = (apr_uint64_t)apr_hash_get(enum_hash->values, value, APR_HASH_KEY_STRING);
		if (!position) {
			enum_hash->poscount++;
			apr_hash_set(enum_hash->values, value, APR_HASH_KEY_STRING, (const void *)enum_hash->poscount);
			PICVIZ_USERDATA(userdata, struct enum_hash_t *) = enum_hash;

			retval = _enum_position_factorize(enum_hash->poscount);
		} else {
			retval = _enum_position_factorize(position);
		}


		enum_hash->poscount++;
		PICVIZ_USERDATA(userdata, struct enum_hash_t *) = enum_hash;


	}

	if (val_len > 10) {
		value[10] = store_c_10;
	}

	return retval;
}

LibCPPExport int picviz_mapping_init()
{
	return 0;
}

LibCPPExport int picviz_mapping_terminate()
{
	return 0;
}

LibCPPExport int picviz_mapping_test()
{
	return 0;
}

