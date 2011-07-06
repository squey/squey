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
#include <picviz/limits.h>
#include <picviz/PVMapping.h>

int first_enum_index = -1;

int string_is_ipv4(const char *string)
{
	int i;

	int cannot_be_ipv4 = 0;

	if (isdigit(string[0])) {
		if (strlen(string) > 15) {
			return 0;
		}
		if (strlen(string) > 5) {
			for (i=0;i<5;i++) {
				if (string[i] > 64) {
			  		cannot_be_ipv4 = 1;
				}
	    		}
		} else {
			return 0;
		}
	} else {
		return 0;
	}

	if (cannot_be_ipv4) {
		return 0;
	}
	
	return 1;
}


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

	char *buffer = NULL;
	uint32_t intval = 0;
	int buf_val;
	int count = 1;

	int i = 0;

	picviz_mapping_t *mapping_o = (picviz_mapping_t *)mapping;


	/* convert_ret = picviz_string_toi(value, &converted); */

	convert_ret = string_is_ipv4(value);

	if (!convert_ret) {
	  /* Retval is not 0, then it cannot be converted as integer */
		if (index != first_enum_index) {
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
		if (!value) {
			picviz_debug(PICVIZ_DEBUG_CRITICAL, "%s: Cannot get the string (NULL) value! Returns 0\n", __FUNCTION__);
			return 0;
		}

		buffer = strdup(value);

		buffer = strtok(buffer, ".");
		if (!buffer) {
			picviz_debug(PICVIZ_DEBUG_CRITICAL, "%s: Cannot get the buffer (%s) value! Returns 0\n", __FUNCTION__, value);
			return 0;
		}
		intval = atoi(buffer) << 24;
		while(buffer) {
			buffer = strtok(NULL, ".");
			if ((!buffer)&&(count == 4)) break;
			if (!buffer) {
				picviz_debug(PICVIZ_DEBUG_CRITICAL, "%s: Cannot get the second buffer (%s) value! Returns 0\n", __FUNCTION__, value);
				return 0;
			}
			buf_val = atoi(buffer);
			count++;
		}
		free(buffer);

		retval = (float)intval;
		retval = retval / PICVIZ_IPV4_MAXVAL;
		retval = retval / 2;		
		
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

