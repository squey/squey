#ifndef PVHIVE_PVOBSERVERBASE_H
#define PVHIVE_PVOBSERVERBASE_H

namespace PVHive {

class PVHive;

/**
 * @class PVObserverBase
 *
 * A non template class to use as a base in the PVHive.
 */
class PVObserverObjectBase
{
public:
	friend class PVHive;

public:
	PVObserverObjectBase() : _object(nullptr) {}
	virtual ~PVObserverObjectBase() { };

protected:
	void *get_object() const
	{
		return _object;
	}

	void set_object(void *object)
	{
		_object = object;
	}

protected:
	void* _object;
};

}

#endif
