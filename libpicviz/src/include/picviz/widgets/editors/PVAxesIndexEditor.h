//! \file PVAxisIndexEditor.h
//! $Id: PVAxisIndexEditor.h 2498 2011-04-25 14:27:23Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVAXESINDEXEDITOR_H
#define PVCORE_PVAXESINDEXEDITOR_H

#include <QListWidget>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/widgets/PVSizeHintListWidget.h>

#include <picviz/PVView.h>

namespace PVWidgets {

/**
 * \class PVAxesIndexEditor
 */
class PVAxesIndexEditor : public PVWidgets::PVSizeHintListWidget<>
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVAxesIndexType _axes_index READ get_axes_index WRITE set_axes_index USER true)

public:
	PVAxesIndexEditor(Picviz::PVView const& view, QWidget *parent = 0);
	virtual ~PVAxesIndexEditor();

public:
	PVCore::PVAxesIndexType get_axes_index() const;
	void set_axes_index(PVCore::PVAxesIndexType axes_index);

protected:
	Picviz::PVView const& _view;
};

}

#endif
