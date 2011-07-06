#ifndef PVCORE_PVREGISTRABLE_CLASS
#define PVCORE_PVREGISTRABLE_CLASS

#include <pvcore/general.h>
#include <boost/shared_ptr.hpp>

namespace PVCore {

template <typename RegAs_>
class LibExport PVRegistrableClass
{
public:
	typedef RegAs_ RegAs;
	typedef boost::shared_ptr< PVRegistrableClass<RegAs_> > p_type;
public:
	template <typename Tc>
	boost::shared_ptr<Tc> clone() const { return boost::shared_ptr<Tc>((Tc*) _clone_me()); }
protected:
	virtual void* _clone_me() const = 0;
};

}

#define CLASS_REGISTRABLE(T) \
	public:\
		typedef boost::shared_ptr<T> p_type;\
	protected:\
		virtual void* _clone_me() const { return new T(*this); }\

#endif
