#ifndef PVCORE_PVFUNCTION_ARGS_H
#define PVCORE_PVFUNCTION_ARGS_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>

#include <QString>

#include <exception>

namespace PVCore {

/*! \brief Exception class if a filter argument is missing during PVFilterFunction::set_args()
 */
class LibKernelDecl PVFunctionArgumentMissing : public std::exception
{
public:
	PVFunctionArgumentMissing(QString const& arg) throw() :
		std::exception()
	{
		_what = QString("Argument %1 missing").arg(arg);
	}
	~PVFunctionArgumentMissing() throw() {};
	virtual const char* what() const throw() { return qPrintable(_what); };
protected:
	QString _what;
};

template <class F>
class PVFunctionArgs
{
public:
	typedef F func_type;
public:
	PVFunctionArgs(PVArgumentList const& args = PVFunctionArgs<F>::default_args()) :
		_args(args), _def_args(args)
	{
	}

	virtual ~PVFunctionArgs() { }
public:
	static PVArgumentList default_args() { return PVArgumentList(); }
public:
	virtual func_type f() = 0;
	virtual const PVArgumentList& get_args() const { return _args; }
	virtual void set_args(PVArgumentList const& args)
	{
		PVArgumentList::const_iterator it,ite;
		it = _def_args.begin();
		ite = _def_args.end();
		for (; it != ite; it++) {
			// If that default argument is not present in the given list
			if (args.find(it.key()) == args.end()) {
				// An exception is thrown
				throw PVFunctionArgumentMissing(it.key());
			}
		}
	   _args = args;
	}
	PVArgumentList const& get_default_args() { return _def_args; }
protected:
	PVArgumentList _args;
	PVArgumentList _def_args;
};

}

#define DEFAULT_ARGS_FUNC(T)\
	PVCore::PVArgumentList T::default_args()

#define CLASS_FUNC_ARGS_NOPARAM(T)\
	public:\
		static PVCore::PVArgumentList default_args() { return PVCore::PVArgumentList(); }

#define CLASS_FUNC_ARGS_PARAM(T)\
	public:\
		static PVCore::PVArgumentList default_args();\

#endif
