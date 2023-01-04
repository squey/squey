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

#ifndef PVCORE_PVPLAINTEXTTYPE_H
#define PVCORE_PVPLAINTEXTTYPE_H

#include <pvkernel/core/PVArgument.h>

#include <QString>
#include <QMetaType>

namespace PVCore
{

class PVPlainTextType : public PVArgumentType<PVPlainTextType>
{
  public:
	PVPlainTextType() { set_text(""); }
	explicit PVPlainTextType(QString const& txt) { set_text(txt); }

	inline void set_text(QString const& txt) { _txt = txt; }
	inline QString const& get_text() const { return _txt; }

	QString to_string() const override { return _txt; }
	PVArgument from_string(QString const& str, bool* ok /*= 0*/) const override
	{
		PVArgument arg;
		arg.setValue(PVPlainTextType(str));

		if (ok) {
			*ok = true;
		}

		return arg;
	}
	bool operator==(const PVPlainTextType& other) const { return _txt == other._txt; }

  private:
	QString _txt;
};
} // namespace PVCore

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(PVCore::PVPlainTextType)

#endif
