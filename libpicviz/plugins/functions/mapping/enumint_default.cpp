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

int first_enum_index = -1;

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
struct enumint_hash_t {
	apr_hash_t *values;
	int poscount;
};
        struct enumint_hash_t *enumint_hash;

	float retval;
	apr_uint64_t position = 0;

	int convert_ret;
	int converted;

	picviz_mapping_t *mapping_o = (picviz_mapping_t *)mapping;


	converted = strtol(value, (char **)NULL, 10);
	if (!isdigit(value[0])) {
	  /* Retval is not 0, then it cannot be converted as integer */
		if (first_enum_index != index) {
			first_enum_index = index;
			enumint_hash = (struct enumint_hash_t *)malloc(sizeof(struct enumint_hash_t));
			enumint_hash->values = apr_hash_make(mapping_o->pool);
			enumint_hash->poscount = 1;
			apr_hash_set(enumint_hash->values, value, APR_HASH_KEY_STRING, (const void *)enumint_hash->poscount);
			PICVIZ_USERDATA(userdata, struct enumint_hash_t *) = enumint_hash;
			retval = _enum_position_factorize(enumint_hash->poscount);
		} else {
			enumint_hash = PICVIZ_USERDATA(userdata, struct enumint_hash_t *);
			position = (apr_uint64_t)apr_hash_get(enumint_hash->values, value, APR_HASH_KEY_STRING);
			if (!position) {
				enumint_hash->poscount++;
				apr_hash_set(enumint_hash->values, value, APR_HASH_KEY_STRING, (const void *)enumint_hash->poscount);
				PICVIZ_USERDATA(userdata, struct enumint_hash_t *) = enumint_hash;

				retval = _enum_position_factorize(enumint_hash->poscount);
			} else {
				retval = _enum_position_factorize(position);
			}

			enumint_hash->poscount++;
			PICVIZ_USERDATA(userdata, struct enumint_hash_t *) = enumint_hash;
		}

		retval = (retval / 2) + 0.5;
		return retval;
	} else {
		retval = (float)converted;
		retval = retval / 65535;
		retval = (retval / 2);
		return retval;
	}

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

