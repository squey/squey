#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifdef WIN32
#include <float.h> // for _logb()
#endif

#include <QHash>
#include <QMutex>
#include <QString>

#include <picviz/general.h>

#include <picviz/PVMapping.h>

#include <iostream>

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
	
	res = res / (float)pow((float)2, (int)N+1);
	
	return res;
}

/* Add is_last to deallocate the hash! */
LibCPPExport float picviz_mapping_exec(const Picviz::PVMapping_p mapping, PVCol index, QString &value, void*, bool is_first)
{
	float retval = 0;
	int position = 0;

	struct enum_hash_t {
		QHash<QString, int> values;
		int poscount;
	};
	static struct enum_hash_t enum_hash[PICVIZ_AXES_MAX];

	static QMutex lock;
	if (is_first) {
		lock.lock();
		enum_hash[index].poscount = 1;
		enum_hash[index].values.clear();
		enum_hash[index].values.insert(value, 1);
		lock.unlock();
		retval = _enum_position_factorize(1);
	} else {
		lock.lock();
		const bool contain = enum_hash[index].values.contains(value);
		lock.unlock();
		if (contain) {
			position = enum_hash[index].values.value(value);
			retval = _enum_position_factorize(position);
		} else {
			lock.lock();
			enum_hash[index].poscount++;
			enum_hash[index].values.insert(value, enum_hash[index].poscount);
			lock.unlock();
			retval = _enum_position_factorize(enum_hash[index].poscount);
		}
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

