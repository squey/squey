/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#ifndef __PVERFPARAMSWIDGET_H__
#define __PVERFPARAMSWIDGET_H__

#include <QDialog>
#include <QDomDocument>

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
	rapidjson::Document get_selected_nodes() const;

  private:
	std::unique_ptr<PVRush::PVERFTreeModel> _model;
};

} // namespace PVRush

#endif // __PVERFPARAMSWIDGET_H__
