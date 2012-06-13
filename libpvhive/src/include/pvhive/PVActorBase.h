#ifndef LIVPVHIVE_PVACTORBASE_H_
#define LIVPVHIVE_PVACTORBASE_H_

namespace PVHive
{
class PVHive;

class PVActorBase
{
public:
	friend class PVHive;
public:
	PVActorBase() : _object(nullptr) {}
	virtual ~PVActorBase();

protected:
	void* _object;
};

}

#endif /* LIVPVHIVE_PVACTORBASE_H_ */
