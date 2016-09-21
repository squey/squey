/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVZONEINDEXEDITOR_H
#define PVCORE_PVZONEINDEXEDITOR_H

#include <QComboBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/PVZoneIndexType.h>

#include <inendi/PVView.h>

namespace PVWidgets
{

/**
 * \class PVZoneIndexEditor
 */
class PVZoneIndexEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(
	    PVCore::PVZoneIndexType _zone_index READ get_zone_index WRITE set_zone_index USER true)

  public:
	explicit PVZoneIndexEditor(Inendi::PVView const& view, QWidget* parent = 0);
	~PVZoneIndexEditor() override;

  public:
	PVCore::PVZoneIndexType get_zone_index() const;
	void set_zone_index(PVCore::PVZoneIndexType zone_index);

  protected:
	Inendi::PVView const& _view;
};
} // namespace PVWidgets

#endif // PVCORE_PVZONEINDEXEDITOR_H
