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

#ifndef PVCORE_PVENUMTYPE_H
#define PVCORE_PVENUMTYPE_H

#include <QMetaType>
#include <QString>
#include <QStringList>
#include <assert.h>

#include <pvkernel/core/PVArgument.h>

namespace PVCore
{

/**
 * \class PVEnumType
 *
 * \note This class is fully implemented in its definition, so no is needed (each library will have
 *its own version).
 */
class PVEnumType : public PVArgumentType<PVEnumType>
{

  public:
	/**
	 * Constructor
	 */
	PVEnumType() { _sel = -1; };
	PVEnumType(QStringList const& list, int sel)
	{
		_list = list;
		_sel = sel;
	};

	QStringList const& get_list() const { return _list; }
	QString get_sel() const
	{
		assert(_sel != -1);
		return _list[_sel];
	}
	int get_sel_index() const { return _sel; }
	void set_sel(int index)
	{
		assert(index < _list.count() && index >= 0);
		_sel = index;
	};
	bool set_sel_from_str(QString const& s)
	{
		int r = _list.indexOf(s);
		if (r == -1)
			return false;
		_sel = r;
		return true;
	}

	QString to_string() const override { return QString::number(_sel); }
	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const override
	{
		bool res_ok = false;

		PVArgument arg;
		arg.setValue(PVEnumType(_list, str.toInt(&res_ok)));

		if (ok) {
			*ok = res_ok;
		}

		return arg;
	}
	bool operator==(const PVEnumType& other) const
	{
		return _list == other._list && _sel == other._sel;
	}

  protected:
	QStringList _list;
	int _sel;
};
} // namespace PVCore

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVEnumType)

#endif
