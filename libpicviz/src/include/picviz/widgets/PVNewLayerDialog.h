/**
 * \file PVNewLayerDialog.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVGUIQT_PVNEWLAYERDIALOG_H__
#define __PVGUIQT_PVNEWLAYERDIALOG_H__

#include <QCheckBox>
#include <QDialog>
#include <QLineEdit>
#include <QMainWindow>
#include <QString>
#include <QWidget>

namespace PVWidgets {

class PVNewLayerDialog : public QDialog
{
	Q_OBJECT;

public:
	static QString get_new_layer_name_from_dialog(const QString& layer_name, bool& hide_layers);

private:
	PVNewLayerDialog(const QString& layer_name, bool hide_layers, QWidget* parent = 0);
	QString get_layer_name() const;
	bool should_hide_layers() const;

private:
	QLineEdit* _text;
	QCheckBox* _checkbox;
};

}

#endif // __PVGUIQT_PVNEWLAYERDIALOG_H__
