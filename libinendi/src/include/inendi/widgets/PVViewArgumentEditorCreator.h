/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVARGUMENTEDITORCREATOR_H
#define PVARGUMENTEDITORCREATOR_H

#include <QByteArray>
#include <QItemEditorCreatorBase>
#include <QWidget>

#include <inendi/PVView.h>

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
