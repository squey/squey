
#ifndef LIVPVHIVE_HIVE_H
#define LIVPVHIVE_HIVE_H

namespace PVHive
{

class PVHive
{
public:
	PVHive &get()
	{
		if (_hive == nullptr) {
			_hive = new PVHive;
		}
		return *_hive;
	}
private:
	static PVHive *_hive;
};

}

#endif // LIVPVHIVE_HIVE_H
