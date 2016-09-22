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
	explicit inline PVViewArgumentEditorCreator(Inendi::PVView const& view)
	    : propertyName(T::staticMetaObject.userProperty().name()), _view(view)
	{
	}
	inline QWidget* createWidget(QWidget* parent) const override { return new T(_view, parent); }
	inline QByteArray valuePropertyName() const override { return propertyName; }

  private:
	QByteArray propertyName;
	Inendi::PVView const& _view;
};
} // namespace PVWidgets

#endif /* PVARGUMENTEDITORCREATOR_H */
