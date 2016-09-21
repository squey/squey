/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVSPINBOXEDITOR_H
#define PVCORE_PVSPINBOXEDITOR_H

#include <QSpinBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/PVSpinBoxType.h>

namespace Inendi
{
class PVView;
}

namespace PVWidgets
{

class PVViewRowsSpinBoxEditor : public QSpinBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVSpinBoxType _s READ get_spin WRITE set_spin USER true)

  public:
	explicit PVViewRowsSpinBoxEditor(Inendi::PVView const& view, QWidget* parent = 0);
	virtual ~PVViewRowsSpinBoxEditor();

  public:
	PVCore::PVSpinBoxType get_spin() const;
	void set_spin(PVCore::PVSpinBoxType s);

  protected:
	PVCore::PVSpinBoxType _s;
	Inendi::PVView const& _view;
};
}

#endif // PVCORE_PVSPINBOXEDITOR_H
