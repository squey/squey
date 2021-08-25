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

#ifndef PVCORE_PVAXISINDEXCHECKBOXTYPE_H
#define PVCORE_PVAXISINDEXCHECKBOXTYPE_H

#include <pvbase/types.h>

#include <pvkernel/core/PVArgument.h>

#include <QMetaType>
#include <QString>
#include <QStringList>

namespace PVCore
{

/**
 * \class PVAxisIndexCheckBoxType
 */
class PVAxisIndexCheckBoxType : public PVArgumentType<PVAxisIndexCheckBoxType>
{

  public:
	/**
	 * Constructor
	 */
	PVAxisIndexCheckBoxType();
	PVAxisIndexCheckBoxType(PVCol origin_axis_index, bool is_checked);

	inline PVCol get_original_index() { return _origin_axis_index; }

	inline bool get_checked() { return _is_checked; }
	inline void set_checked(const bool checked) { _is_checked = checked; }

	QString to_string() const override
	{
		return QString::number(_origin_axis_index) + ":" + QString(_is_checked ? "true" : "false");
	}
	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const override
	{
		PVArgument arg;
		bool res_ok = false;

		QStringList parts = str.split(":");
		if (parts.count() == 2) {
			PVCol origin_axis_index = (PVCol)parts[0].toInt(&res_ok);
			bool is_checked = parts[1].compare("true", Qt::CaseInsensitive) == 0;
			arg.setValue(PVAxisIndexCheckBoxType(origin_axis_index, is_checked));
		}

		if (ok) {
			*ok = res_ok;
		}

		return arg;
	}
	bool operator==(const PVAxisIndexCheckBoxType& other) const
	{
		return _origin_axis_index == other._origin_axis_index && _is_checked == other._is_checked;
	}

  protected:
	// The original axis index will never change. PVAxisCombination takes care of any
	// axis addition/order modification, but will never change the original axis index.
	PVCol _origin_axis_index;
	bool _is_checked;
};
} // namespace PVCore

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVAxisIndexCheckBoxType)

#endif // PVCORE_PVAXISINDEXCHECKBOXTYPE_H
