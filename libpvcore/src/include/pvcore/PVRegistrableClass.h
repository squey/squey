#ifndef PVCORE_PVREGISTRABLE_CLASS
#define PVCORE_PVREGISTRABLE_CLASS

#include <pvcore/general.h>
#include <boost/shared_ptr.hpp>

namespace PVCore {

// Forward declaration
template <class RegAs> class PVClassLibrary;

template <typename RegAs_>
class PVRegistrableClass
{
	template <class RegAs>
	friend class PVCore::PVClassLibrary;
public:
	typedef RegAs_ RegAs;
	typedef boost::shared_ptr< PVRegistrableClass<RegAs_> > p_type;
public:
	template <typename Tc>
	boost::shared_ptr<Tc> clone() const
	{
		PVRegistrableClass* rc = _clone_me();
		assert(rc);
		rc->__registered_class_name = __registered_class_name;
		Tc* ret = dynamic_cast<Tc*>(rc);
		assert(ret);
		return boost::shared_ptr<Tc>(ret);
	}
	QString const& registered_name() { return __registered_class_name; }
protected:
	virtual PVRegistrableClass* _clone_me() const = 0;
protected:
	QString __registered_class_name;
};

}

#define CLASS_REGISTRABLE(T) \
	public:\
		typedef boost::shared_ptr<T> p_type;\
	protected:\
		virtual PVCore::PVRegistrableClass* _clone_me() const { T* ret = new T(*this); return ret; }\

#define CLASS_REGISTRABLE_NOCOPY(T) \
	public:\
		typedef boost::shared_ptr<T> p_type;\
	protected:\
		virtual PVCore::PVRegistrableClass* _clone_me() const { T* ret = new T(); return ret; }\

#endif
