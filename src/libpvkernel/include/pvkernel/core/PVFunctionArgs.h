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

#ifndef PVCORE_PVFUNCTION_ARGS_H
#define PVCORE_PVFUNCTION_ARGS_H

#include <pvkernel/core/PVArgument.h>

#include <QString>

#include <exception>

namespace PVCore
{

/*! \brief Exception class if a filter argument is missing during PVFilterFunction::set_args()
 */
class PVFunctionArgumentMissing : public std::exception
{
  public:
	explicit PVFunctionArgumentMissing(QString const& arg) throw() : std::exception()
	{
		_what = QString("Argument %1 missing").arg(arg);
	}
	~PVFunctionArgumentMissing() throw() override = default;
	;
	const char* what() const throw() override { return qPrintable(_what); };

  protected:
	QString _what;
};

class PVFunctionArgsBase
{

  public:
	explicit PVFunctionArgsBase(PVArgumentList const& args = PVArgumentList())
	    : _args(args), _def_args(args)
	{
	}

	virtual ~PVFunctionArgsBase() = default;

  public:
	virtual const PVArgumentList& get_args() const { return _args; }
	virtual void set_args(PVArgumentList const& args)
	{
		PVArgumentList::const_iterator it, ite;
		it = _def_args.begin();
		ite = _def_args.end();
		for (; it != ite; it++) {
			// If that default argument is not present in the given list
			if (args.find(it->key()) == args.end()) {
				// An exception is thrown
				throw PVFunctionArgumentMissing(it->key());
			}
		}
		_args = args;
	}

	PVArgumentList const& get_default_args() const { return _def_args; }

	PVArgumentList get_args_for_preset() const
	{
		PVArgumentList args = get_args();
		PVCore::PVArgumentKeyList keys = get_args_keys_for_preset();

		// Get rid of unwanted args
		PVArgumentList filtered_args;
		for (PVCore::PVArgumentKey key : keys) {
			PVArgumentList::const_iterator it = args.find(key);
			if (it != args.end()) {
				filtered_args[it->key()] = it->value();
			}
		}

		return filtered_args;
	}
	virtual PVCore::PVArgumentKeyList get_args_keys_for_preset() const
	{
		return get_default_args().keys();
	}
	void set_args_from_preset(PVArgumentList const& args)
	{
		PVArgumentList preset_args = get_args_for_preset();
		PVArgumentList::const_iterator it;
		for (it = args.begin(); it != args.end(); it++) {
			if (preset_args.contains(it->key())) {
				_args[it->key()] = it->value();
			}
		}
	}

  protected:
	PVArgumentList _args;
	PVArgumentList _def_args;
};

// FIXME: is this really useful ?!
template <class F>
class PVFunctionArgs : public PVFunctionArgsBase
{
  public:
	explicit PVFunctionArgs(PVArgumentList const& args = PVArgumentList())
	    : PVFunctionArgsBase(args)
	{
	}
};
} // namespace PVCore

#define DEFAULT_ARGS_FUNC(T) PVCore::PVArgumentList T::default_args()

#define CLASS_FUNC_ARGS_NOPARAM(T)                                                                 \
  public:                                                                                          \
	static PVCore::PVArgumentList default_args() { return PVCore::PVArgumentList(); }

#define CLASS_FUNC_ARGS_PARAM(T)                                                                   \
  public:                                                                                          \
	static PVCore::PVArgumentList default_args();

#endif
