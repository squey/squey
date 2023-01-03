/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVFILTER_PVFILTERFUNCTION_H
#define PVFILTER_PVFILTERFUNCTION_H

#include <pvkernel/core/PVFunctionArgs.h>
#include <pvkernel/core/PVRegistrableClass.h>

#include <boost/function.hpp>

#include <memory>

namespace PVFilter
{

//! \brief Base class of a filter function
//! \tparam Tout the type of output object
//! \tparam Tin type type on input object
//!
//! This class define the filter function model. This model is used in libpvfilter and other
//! libraries to define filters.\n
//! Each filter as an input (Tin) and output (Tout) type, define as template parameters.
//! Every filter class must overload PVFilterFunctionBase (or one of its child). This base class is
// responsible for:
//! <ul>
//! <li>defining the interface of the filter (for instance the functor operator) according to the
// input/output type of the filter.</li>
//! <li>managing the arguments of the filter (through a PVArgumentList object), and checking that no
// argument is missing (see PVFilterFunctionBase::set_args)</li>
//! <li>defining common types for a filter, like "p_type" (shared pointer type of the object) or
//"func_type" (boost::function bound object that represent the filter)</li>
//! </ul>
//! See PVFilterFunctionRegistrable for more details.
//!
//! In order to create a new registrable filter, you need to:
//! <ul>
//! <li>subclass PVFilterFunctionRegistrable (which is a child of PVFilterFunctionBase, and allows
// the filter to be registered by PVFilterLibrary, by implementing the "_clone_me" protected
// method)</li>
//! <li>in your class constructor, you need to call INIT_FILTER(NameOfYourClass, args) (where "args"
// is a PVArgumentList provided in your constructor function arguments) or
// INIT_FILTER_NOPARAM(NameOfYourClass) (if you have no arguments).</li>
//! <li>in the case your filter needs arguments, you must define them using the
// DEFAULT_ARGS_FILTER(NameOfYourClassWithNamespace) macro (see example below)</li>
//! <li>you also need to insert a "CLASS_FILTER(NameOfYourClass)" macro inside your class
// definition</li>
//! </ul>
//!
//! Here is an example of how to implement a filter that convert a QString to const char* using
// system locale (should not be used in real life !) :
//! \code
//! // In the header file:
//! namespace Inendi
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
//! Inendi::PVFilterQString::PVFilterQString(PVCore::PVArgumentList const& args)
//! {
//! 	INIT_FILTER(PVFilterQString, args);
//! }
//!
//! DEFAULT_ARGS_FILTER(Inendi::PVFilterQString)
//! {
//! 	PVCore::PVArgumentList args;
//! 	args["arg1"] = true; // This is a QVariant !
//! }
//! \endcode
//
//! As said above, a filter function takes two templates argument: Tin and Tout. It specifies that
// this filter takes
//! a Tin object as input, and renders a Tout object as output. The "functor" operator is used for
// this
//! purpose, and is overloaded by child classes.
//! \code
//! virtual Tout operator()(Tin obj) = 0; // This is overloaded by child classes
//! \endcode
//!
//! Arguments are defined through a PVArgumentList object. The "INIT_FILTER(T)" macro used in a
// filter constructor create the PVArgumentList _def_args objects, filled with
//! the default arguments provided by the static function T::default_args().
//! When arguments of a filter are set using the "set_args" function, the keys are compared against
// the one of _def_args, and it checks that none are missing.
//! Moreover, three types are defined for a filter (using typedef):
//!
//! <ul>
//! <li>func_type, that defines the boost::function type returned by the f() function. Defined by
// PVFilterFunctionBase and must *never* be overrided</li>
//! <li>p_type, that *must* defines a shared pointer to the current class (which means that is
// overrided by the CLASS_FILTER macro to match the corresponding type)</li>
//! <li>base, that defines the base class (PVFilterFunctionBase<Tout,Tin). This must *never* be
// overrided.</li>
//! </ul>
template <typename Tout_, typename Tin_>
class PVFilterFunctionBase : public PVCore::PVFunctionArgs<boost::function<Tout_(Tin_)>>
{
  public:
	typedef Tout_ Tout;
	typedef Tin_ Tin;
	typedef boost::function<Tout(Tin)> func_type;
	typedef std::shared_ptr<PVFilterFunctionBase<Tout, Tin>> p_type;
	typedef PVFilterFunctionBase<Tout, Tin> base;

  public:
	explicit PVFilterFunctionBase(
	    PVCore::PVArgumentList const& args = PVFilterFunctionBase<Tout, Tin>::default_args())
	    : PVCore::PVFunctionArgs<func_type>(args)
	{
	}

  public:
	static PVCore::PVArgumentList default_args() { return PVCore::PVArgumentList(); }

  public:
	QString const& get_name() const { return _name; }

  protected:
	QString _name;
};

/*! \brief Special PVFilterFunctionBase function for void -> Tin
 */
template <typename Tout_>
class PVFilterFunctionBase<Tout_, void> : public PVCore::PVFunctionArgs<boost::function<Tout_()>>
{
  public:
	typedef Tout_ Tout;
	typedef void Tin;
	typedef boost::function<Tout()> func_type;
	typedef std::shared_ptr<PVFilterFunctionBase<Tout, void>> p_type;
	typedef PVFilterFunctionBase<Tout, void> base;

  public:
	explicit PVFilterFunctionBase(
	    PVCore::PVArgumentList const& args = PVFilterFunctionBase<Tout, void>::default_args())
	    : PVCore::PVFunctionArgs<func_type>(args)
	{
	}

  public:
	static PVCore::PVArgumentList default_args() { return PVCore::PVArgumentList(); }

  public:
	QString const& get_name() { return _name; }

  protected:
	QString _name;
};

/*! \brief Define a filter function that takes the same type as reference in input and output (Tout
 *= T&, Tin = T&)
 *
 * Define a filter function that takes the same type as reference in input and output (Tout = T&,
 *Tin = T&) and that is registrable.\n
 * Used by many filters in libinendi and others.
 */
template <typename T, typename FilterT_ = PVFilterFunctionBase<T&, T&>>
class PVFilterFunction : public PVFilterFunctionBase<T&, T&>,
                         public PVCore::PVRegistrableClass<FilterT_>
{
  public:
	typedef FilterT_ FilterT;
	typedef FilterT RegAs;
	typedef std::shared_ptr<PVFilterFunction<T, FilterT>> p_type;
	// typedef PVFilterFunction<T,FilterT_> base_registrable;
	typedef typename PVFilterFunctionBase<T&, T&>::func_type func_type;

  public:
	explicit PVFilterFunction(PVCore::PVArgumentList const& args = PVFilterFunction::default_args())
	    : PVFilterFunctionBase<T&, T&>(args), PVCore::PVRegistrableClass<RegAs>()
	{
	}

  public:
	static PVCore::PVArgumentList default_args() { return PVCore::PVArgumentList(); }

  public:
	virtual T& operator()(T& obj) = 0;
};
} // namespace PVFilter

// Macros for filter class construction help
#define CLASS_FILTER(T)                                                                            \
	CLASS_FILTER_NONREG(T)                                                                         \
	CLASS_REGISTRABLE(T)

#define CLASS_FILTER_NOPARAM(T)                                                                    \
	CLASS_FILTER_NONREG_NOPARAM(T)                                                                 \
	CLASS_REGISTRABLE(T)

#define CLASS_FILTER_NONREG(T)                                                                     \
  public:                                                                                          \
	CLASS_FUNC_ARGS_PARAM(T)

#define CLASS_FILTER_NONREG_NOPARAM(T)                                                             \
  public:                                                                                          \
	CLASS_FUNC_ARGS_NOPARAM(T)

#define INIT_FILTER(T, aparams)                                                                    \
	do {                                                                                           \
		_def_args = T::default_args();                                                             \
		set_args((aparams));                                                                       \
	} while (0)

#define INIT_FILTER_NOPARAM(T)                                                                     \
	do {                                                                                           \
		_def_args = PVCore::PVArgumentList();                                                      \
		_args = _def_args;                                                                         \
	} while (0)

#define DEFAULT_ARGS_FILTER(T) DEFAULT_ARGS_FUNC(T)

#endif
