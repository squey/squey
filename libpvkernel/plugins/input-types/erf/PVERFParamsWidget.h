/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#ifndef __PVERFPARAMSWIDGET_H__
#define __PVERFPARAMSWIDGET_H__

#include <QDialog>

#include <pvkernel/rush/PVFormat.h>

#include "PVERFTreeModel.h"

namespace PVRush
{

class PVInputTypeERF;

class PVERFParamsWidget : public QDialog
{
	Q_OBJECT

  public:
	PVERFParamsWidget(PVInputTypeERF const* in_t, QWidget* parent);

  public:
	std::vector<QDomDocument> get_formats();
	QString path() const;
	std::vector<std::tuple<rapidjson::Document, std::string, PVRush::PVFormat>>
	get_sources_info() const;

  private:
	std::unique_ptr<PVRush::PVERFTreeModel> _model;
};

} // namespace PVRush

#endif // __PVERFPARAMSWIDGET_H__
