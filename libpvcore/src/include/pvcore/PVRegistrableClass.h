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
	boost::shared_ptr<Tc> clone() const { return boost::shared_ptr<Tc>((Tc*) _clone_me()); }
    template <typename Tc>
	boost::shared_ptr<Tc> new_object() const { return boost::shared_ptr<Tc>((Tc*) _new_object()); }
	QString const& registered_name() { return __registered_class_name; }
protected:
	virtual void* _clone_me() const = 0;
    virtual void* _new_object() const = 0;
protected:
	QString __registered_class_name;
};

}

#define CLASS_REGISTRABLE(T) \
	public:\
		typedef boost::shared_ptr<T> p_type;\
	protected:\
		virtual void* _clone_me() const { T* ret = new T(*this); return ret; }\
        virtual void* _new_object() const { T* ret = new T(); ret->__registered_class_name = __registered_class_name; return ret; }\

#endif
