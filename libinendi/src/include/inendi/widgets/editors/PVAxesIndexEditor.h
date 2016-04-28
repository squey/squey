/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVAXESINDEXEDITOR_H
#define PVCORE_PVAXESINDEXEDITOR_H

#include <QListWidget>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/widgets/PVSizeHintListWidget.h>

#include <inendi/PVView.h>

namespace PVWidgets
{

/**
 * \class PVAxesIndexEditor
 */
class PVAxesIndexEditor : public PVWidgets::PVSizeHintListWidget<>
{
	Q_OBJECT
	Q_PROPERTY(
	    PVCore::PVAxesIndexType _axes_index READ get_axes_index WRITE set_axes_index USER true)

  public:
	PVAxesIndexEditor(Inendi::PVView const& view, QWidget* parent = 0);
	virtual ~PVAxesIndexEditor();

  public:
	PVCore::PVAxesIndexType get_axes_index() const;
	void set_axes_index(PVCore::PVAxesIndexType axes_index);

  protected:
	Inendi::PVView const& _view;
};
}

#endif
