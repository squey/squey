
#ifndef LIBPVHIVE_PVOBSERVER_H
#define LIBPVHIVE_PVOBSERVER_H

#include <cassert>

namespace PVHive
{

class PVObserverBase
{
protected:
	virtual void refresh() = 0;
	virtual void about_to_be_deleted() = 0;
};

template <class T>
class PVObserver : public PVObserverBase
{
public:
	PVObserver()
	{
		_object = nullptr;
	}

	void set_object(const T *object)
	{
		assert(_object == nullptr);
		_object = object;
	}

protected:

	T const* get_object() const
	{
		return _object;
	}

protected:
	T const* _object;
};

}

#endif // LIBPVHIVE_PVOBSERVER_H
