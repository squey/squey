//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVDBPreviewWidget.h"
#include <QMessageBox>
#include <QSqlError>

PVRush::PVDBPreviewWidget::PVDBPreviewWidget(PVDBInfos const& infos,
                                             QString const& query,
                                             uint32_t nrows,
                                             QDialog* parent)
    : QDialog(parent), _init(false), _infos(infos), _query_str(query), _nrows(nrows)
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
		QMessageBox err(QMessageBox::Critical, tr("Error while previewing..."),
		                tr("Unable to connect to the database: ") + _serv->last_error(),
		                QMessageBox::Ok);
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
	_table_model->setQuery(std::move(sqlq));
	if (_table_model->lastError().isValid()) {
		QMessageBox err(QMessageBox::Critical, tr("Error while previewing..."),
		                tr("Unable to execute query: ") + _table_model->lastError().driverText(),
		                QMessageBox::Ok);
		err.exec();
	}
}
