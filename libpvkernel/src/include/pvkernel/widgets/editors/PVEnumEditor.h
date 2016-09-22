/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVENUMEDITOR_H
#define PVCORE_PVENUMEDITOR_H

#include <pvkernel/core/PVEnumType.h>

#include <QComboBox>

class QWidget;

namespace PVWidgets
{

/**
 * \class PVEnumEditor
 */
class PVEnumEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVEnumType _enum READ get_enum WRITE set_enum USER true)

  public:
	using QComboBox::QComboBox;

  public:
	PVCore::PVEnumType get_enum() const;
	void set_enum(PVCore::PVEnumType e);

  protected:
	PVCore::PVEnumType _e;
};
} // namespace PVWidgets

#endif // PVCORE_PVEnumEDITOR_H
