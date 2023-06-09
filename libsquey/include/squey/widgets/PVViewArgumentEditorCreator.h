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

#ifndef PVARGUMENTEDITORCREATOR_H
#define PVARGUMENTEDITORCREATOR_H

#include <QByteArray>
#include <QItemEditorCreatorBase>
#include <QWidget>

#include <squey/PVView.h>

namespace PVWidgets
{

// Inspired by QStandardItemEditorCreator
// Reuse the Q_PROPERTY macros
template <class T>
class PVViewArgumentEditorCreator : public QItemEditorCreatorBase
{
  public:
	template <class... Args>
	explicit inline PVViewArgumentEditorCreator(Args const&... args)
	    : propertyName(T::staticMetaObject.userProperty().name())
	    , _create([&args...](QWidget* parent) { return new T(args..., parent); })
	{
	}
	inline QWidget* createWidget(QWidget* parent) const override { return _create(parent); }
	inline QByteArray valuePropertyName() const override { return propertyName; }

  private:
	QByteArray propertyName;
	std::function<QWidget*(QWidget*)> _create;
};
} // namespace PVWidgets

#endif /* PVARGUMENTEDITORCREATOR_H */
