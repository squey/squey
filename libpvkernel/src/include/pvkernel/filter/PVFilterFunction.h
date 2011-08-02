//! \file PVFilterFunction.h
//! \brief Base classes for filtering functions. See PVFilterFunctionBase for detailed informations.
//! $Id: PVFilterFunction.h 3165 2011-06-16 05:05:39Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVFILTERFUNCTION_H
#define PVFILTER_PVFILTERFUNCTION_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVRegistrableClass.h>

#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

#include <exception>

namespace PVFilter {

/*! \brief Exception class if a filter argument is missing during PVFilterFunction::set_args()
 */
class LibKernelDecl PVFilterArgumentMissing : public std::exception
{
public:
	PVFilterArgumentMissing(QString const& arg) throw() :
		std::exception()
	{
		_what = QString("Argument %1 missing").arg(arg);
	}
	~PVFilterArgumentMissing() throw() {};
	virtual const char* what() const throw() { return qPrintable(_what); };
protected:
	QString _what;
};

//! \brief Base class of a filter function
//! \tparam Tout the type of output object
//! \tparam Tin type type on input object
//! 
//! This class define the filter function model. This model is used in libpvfilter and other
//! libraries to define filters.\n
//! Each filter as an input (Tin) and output (Tout) type, define as template parameters.
//! Every filter class must overload PVFilterFunctionBase (or one of its child). This base class is responsible for:
//! <ul>
//! <li>defining the interface of the filter (for instance the functor operator) according to the input/output type of the filter.</li>
//! <li>managing the arguments of the filter (through a PVArgumentList object), and checking that no argument is missing (see PVFilterFunctionBase::set_args)</li>
//! <li>defining common types for a filter, like "p_type" (shared pointer type of the object) or "func_type" (boost::function bound object that represent the filter)</li>
//! </ul>
//! See PVFilterFunctionRegistrable for more details.
//! 
//! In order to create a new registrable filter, you need to:
//! <ul>
//! <li>subclass PVFilterFunctionRegistrable (which is a child of PVFilterFunctionBase, and allows the filter to be registered by PVFilterLibrary, by implementing the "_clone_me" protected method)</li>
//! <li>in your class constructor, you need to call INIT_FILTER(NameOfYourClass, args) (where "args" is a PVArgumentList provided in your constructor function arguments) or INIT_FILTER_NOPARAM(NameOfYourClass) (if you have no arguments).</li>
//! <li>in the case your filter needs arguments, you must define them using the DEFAULT_ARGS_FILTER(NameOfYourClassWithNamespace) macro (see example below)</li>
//! <li>you also need to insert a "CLASS_FILTER(NameOfYourClass)" macro inside your class definition, and an "IMPL_FILTER(NameOfYourClassWithNamespace)" at the end of your implementation file (.cpp) (see example below)</li>
//! </ul>
//!
//! Here is an example of how to implement a filter that convert a QString to const char* using system locale (should not be used in real life !) :
//! \code
//! // In the header file:
//! namespace Picviz
//! {
//! 
//! class PVFilterQString: public PVFilterFunctionRegistrable<const char*, QString const&> {
//! public:
//! 	PVFilterQString(PVCore::PVArgumentList const& args = PVFilterQString::default_args());
//! 	virtual const char* operator()(QString const& str) { return str.toLocal8Bit().constData(); }
//!
//! CLASS_FILTER(PVFilterQString); 
//! };
//! 
//! }
//! 
//! // In the implementation file
//! Picviz::PVFilterQString::PVFilterQString(PVCore::PVArgumentList const& args)
//! {
//! 	INIT_FILTER(PVFilterQString, args);
//! }
//! 
//! DEFAULT_ARGS_FILTER(Picviz::PVFilterQString)
//! {
//! 	PVCore::PVArgumentList args;
//! 	args["arg1"] = true; // This is a QVariant !
//! }
//! 
//! IMPL_FILTER(Picviz::PVFilterQString)
//! \endcode 
//
//! As said above, a filter function takes two templates argument: Tin and Tout. It specifies that this filter takes
//! a Tin object as input, and renders a Tout object as output. The "functor" operator is used for this 
//! purpose, and is overloaded by child classes.
//! \code
//! virtual Tout operator()(Tin obj) = 0; // This is overloaded by child classes
//! \endcode
//! 
//! Arguments are defined through a PVArgumentList object. The "INIT_FILTER(T)" macro used in a filter constructor create the PVArgumentList _def_args objects, filled with
//! the default arguments provided by the static function T::default_args().
//! When arguments of a filter are set using the "set_args" function, the keys are compared against the one of _def_args, and it checks that none are missing.
//! Moreover, three types are defined for a filter (using typedef):
//! 
//! <ul>
//! <li>func_type, that defines the boost::function type returned by the f() function. Defined by PVFilterFunctionBase and must *never* be overrided</li>
//! <li>p_type, that *must* defines a shared pointer to the current class (which means that is overrided by the CLASS_FILTER macro to match the corresponding type)</li>
//! <li>base, that defines the base class (PVFilterFunctionBase<Tout,Tin). This must *never* be overrided.</li>
//! </ul>
template <typename Tout, typename Tin>
class PVFilterFunctionBase
{
public:
	typedef boost::function<Tout(Tin)> func_type;
	typedef boost::shared_ptr< PVFilterFunctionBase<Tout,Tin> > p_type;
	typedef PVFilterFunctionBase<Tout,Tin> base;
public:
	PVFilterFunctionBase(PVCore::PVArgumentList const& args = PVFilterFunctionBase<Tout,Tin>::default_args()) :
		_args(args), _def_args(args)
	{
	}
public:
	static PVCore::PVArgumentList default_args() { return PVCore::PVArgumentList(); }
public:

	virtual Tout operator()(Tin obj) = 0;
	func_type f() { return boost::bind<Tout>(&PVFilterFunctionBase<Tout,Tin>::_f, this, _1); }
	Tout _f(Tin obj) { return this->operator()(obj); }
	virtual const PVCore::PVArgumentList& get_args() const { return _args; }
	virtual void set_args(PVCore::PVArgumentList const& args)
	{
		PVCore::PVArgumentList::const_iterator it,ite;
		it = _def_args.begin();
		ite = _def_args.end();
		for (; it != ite; it++) {
			// If that default argument is not present in the given list
			if (args.find(it.key()) == ite) {
				// An exception is thrown
				throw new PVFilterArgumentMissing(it.key());
			}
		}
	   _args = args;
	}
	QString const& get_name() { return _name; }
	PVCore::PVArgumentList const& get_default_args() { return _def_args; }
protected:
	PVCore::PVArgumentList _args;
	PVCore::PVArgumentList _def_args;
	QString _name;
};

/*! \brief Special PVFilterFunctionBase function for Tin -> void
 */
template <typename Tin>
class PVFilterFunctionBase<void,Tin>
{
public:
	typedef boost::function<void(Tin)> func_type;
	typedef boost::shared_ptr< PVFilterFunctionBase<void,Tin> > p_type;
	typedef PVFilterFunctionBase<void,Tin> base;
public:
	PVFilterFunctionBase(PVCore::PVArgumentList const& args = PVFilterFunctionBase<void,Tin>::default_args()) :
		_args(args), _def_args(args)
	{
	}
public:
	static PVCore::PVArgumentList default_args() { return PVCore::PVArgumentList(); }
public:
	virtual void operator()(Tin /*obj*/ ) = 0;
	func_type f() { return boost::bind<void>(&PVFilterFunctionBase<void,Tin>::_f, this, _1); }
	void _f(Tin obj) { this->operator()(obj); }
	QString const& get_name() { return _name; }
	PVCore::PVArgumentList const& get_default_args() { return _def_args; }
	virtual void set_args(PVCore::PVArgumentList const& args)
	{
		PVCore::PVArgumentList::const_iterator it,ite;
		it = _def_args.begin();
		ite = _def_args.end();
		for (; it != ite; it++) {
			// If that default argument is not present in the given list
			if (args.find(it.key()) == ite) {
				// An exception is thrown
				throw new PVFilterArgumentMissing(it.key());
			}
		}
	   _args = args;
	}
protected:
	PVCore::PVArgumentList _args;
	PVCore::PVArgumentList _def_args;
	QString _name;
};

/*! \brief Special PVFilterFunctionBase function for void -> Tin
 */
template <typename Tout>
class PVFilterFunctionBase<Tout,void>
{
public:
	typedef boost::function<Tout()> func_type;
	typedef boost::shared_ptr< PVFilterFunctionBase<Tout,void> > p_type;
	typedef PVFilterFunctionBase<Tout,void> base;
public:
	PVFilterFunctionBase(PVCore::PVArgumentList const& args = PVFilterFunctionBase<Tout,void>::default_args()) :
		_args(args), _def_args(args)
	{
	}
public:
	static PVCore::PVArgumentList default_args() { return PVCore::PVArgumentList(); }
public:
	virtual Tout operator()() = 0;
	/*! \brief Returns a boost::bind object that calls the operator() function of this filter
	 */
	func_type f() { return boost::bind<Tout>(&PVFilterFunctionBase<Tout,void>::_f, this); }

	/*! \brief Intermediate function for boost::bind.
	 * \note This whole process should be optimised because extra-calls are made, and may be a performance bottleneck.
	 */
	Tout _f() { return this->operator()(); }

	QString const& get_name() { return _name; }
	PVCore::PVArgumentList const& get_default_args() { return _def_args; }

	/*! \brief Set the argument of the filter object.
	 * Set the argument of the filter object, and compares its keys against the default ones to see if none are missing.
	 * If it is the case, a PVFilterAgumentMissing exception is thrown.
	 */
	virtual void set_args(PVCore::PVArgumentList const& args)
	{
		PVCore::PVArgumentList::const_iterator it,ite;
		it = _def_args.begin();
		ite = _def_args.end();
		for (; it != ite; it++) {
			// If that default argument is not present in the given list
			if (args.find(it.key()) == ite) {
				// An exception is thrown
				throw new PVFilterArgumentMissing(it.key());
			}
		}
	   _args = args;
	}
protected:
	PVCore::PVArgumentList _args;
	PVCore::PVArgumentList _def_args;
	QString _name;
};

/*! \brief template class for a registrable function with a parent filter type
 *  \tparam FilterT_ class that will be used as default filter type (see below)
 *  \sa PVFilterFunctionBase
 *
 * Plugins that need to be registered to PVFilterLibrary classes need to inherit from this class.
 * The "FilterT" member type defines the filter type of the filter. As instance, PVElementFilterByGrep is registrable with PVElementFilter (as FilterT).
 */

#if 0
// Forward declaration
template <class FilterT>
class PVFilterLibrary;

template <typename Tout, typename Tin, typename FilterT_ = PVFilterFunctionBase<Tout,Tin> >
class PVFilterFunctionRegistrable: public PVFilterFunctionBase<Tout,Tin>
{
	template <class FilterT>
	friend class PVFilterLibrary;
public:
	typedef FilterT_ FilterT;
	typedef boost::shared_ptr< PVFilterFunctionRegistrable<Tout,Tin,FilterT_> > p_type;
	typedef PVFilterFunctionRegistrable<Tout,Tin,FilterT_> base_registrable;
public:
	PVFilterFunctionRegistrable(PVCore::PVArgumentList const& args = PVFilterFunctionBase<Tout,void>::default_args()) :
		PVFilterFunctionBase<Tout,Tin>(args)	
	{
	}
public:
	virtual boost::shared_ptr<base_registrable> clone_basep() const = 0;

	template <typename Tc>
	boost::shared_ptr<Tc> clone() const { return boost::shared_ptr<Tc>((Tc*) _clone_me()); }

	QString registered_name() const { return __registered_name; }
protected:
	/*! \brief virtual method that is implemented by the IMPL_FILTER macro. This is used for polymorphic cloning.
	 *  \return A pointer to a new object of this class, heap-allocated (with the new operator) and using the copy constructor with *this as parameter.
	 *  \sa IMPL_FILTER, clone
	 */
	virtual void* _clone_me() const = 0;
	QString __registered_name;
};
#endif

/*! \brief Define a filter function that takes the same type as reference in input and output (Tout = T&, Tin = T&)
 *
 * Define a filter function that takes the same type as reference in input and output (Tout = T&, Tin = T&) and that is registrable.\n
 * Used by many filters in libpicviz and others.
 */
template < typename T, typename FilterT_ = PVFilterFunctionBase<T&,T&> >
class PVFilterFunction : public PVFilterFunctionBase<T&,T&>, public PVCore::PVRegistrableClass< FilterT_ > {
public:
	typedef FilterT_ FilterT;
	typedef FilterT RegAs;
	typedef boost::shared_ptr< PVFilterFunction<T,FilterT> > p_type;
	typedef PVFilterFunction<T,FilterT_> base_registrable;
public:
	PVFilterFunction(PVCore::PVArgumentList const& args = PVFilterFunction::default_args()) :
		PVFilterFunctionBase<T&,T&>(args),
		PVCore::PVRegistrableClass<RegAs>()
	{
	}
public:
	static PVCore::PVArgumentList default_args() { return PVCore::PVArgumentList(); }
public:
	virtual T& operator()(T& obj) = 0;
};

}

// Macros for filter class construction help
#define CLASS_FILTER(T) \
	public:\
		static PVCore::PVArgumentList default_args();\
	CLASS_REGISTRABLE(T)

#define CLASS_FILTER_INPLACE(T) \
	CLASS_REGISTRABLE(T)

#define CLASS_FILTER_NOPARAM_INPLACE(T) \
	public:\
		static PVCore::PVArgumentList default_args() { return PVCore::PVArgumentList(); }\
	CLASS_REGISTRABLE(T)

#define IMPL_FILTER(T)


#define IMPL_FILTER_NOPARAM(T)\
	PVCore::PVArgumentList T::default_args()\
	{\
		return PVCore::PVArgumentList();\
	}\

#define INIT_FILTER(T,aparams)\
	do {\
		_def_args = T::default_args();\
		set_args((aparams));\
	} while(0)

#define INIT_FILTER_NOPARAM(T)\
	do {\
		_def_args = T::default_args();\
		_args = _def_args;\
	} while(0)

#define DEFAULT_ARGS_FILTER(T)\
	PVCore::PVArgumentList T::default_args()

#define DEFAULT_ARGS_FILTER_INPLACE(T)\
	PVCore::PVArgumentList default_args()

#endif
