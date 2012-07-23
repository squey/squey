/**
 * \file PVDBPreviewWidget.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVDBPreviewWidget.h"
#include <QMessageBox>
#include <QSqlError>

PVRush::PVDBPreviewWidget::PVDBPreviewWidget(PVDBInfos const& infos, QString const& query, uint32_t nrows, QDialog* parent):
	QDialog(parent),
	_init(false),
	_infos(infos),
	_query_str(query),
	_nrows(nrows)
{
	setupUi(this);
	_table_model = new QSqlQueryModel();
	_table_query->setModel(_table_model);
	setWindowTitle("SQL query preview");
}

bool PVRush::PVDBPreviewWidget::init()
{
	if (_init) {
		return true;
	}

	_serv.reset(new PVDBServ(_infos));
	if (!_serv->connect()) {
		QMessageBox err(QMessageBox::Critical, tr("Error while previewing..."), tr("Unable to connect to the database: ") + _serv->last_error(), QMessageBox::Ok);
		err.exec();
		return false;
	}
	_init = true;
	return true;
}

void PVRush::PVDBPreviewWidget::preview()
{
	if (!_init) {
		init();
	}
	PVDBQuery query(_serv, _query_str);
	QSqlQuery sqlq = query.to_query(0, _nrows);
	sqlq.exec();
	_table_model->setQuery(sqlq);
	if (_table_model->lastError().isValid()) {
		QMessageBox err(QMessageBox::Critical, tr("Error while previewing..."), tr("Unable to execute query: ") + _table_model->lastError().driverText(), QMessageBox::Ok);
		err.exec();
	}
}
