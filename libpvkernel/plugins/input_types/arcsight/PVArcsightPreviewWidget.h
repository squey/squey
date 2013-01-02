#ifndef PVDBPREVIEWWIDGET_FILE_H
#define PVDBPREVIEWWIDGET_FILE_H

#include "../ui_db_preview.h"
#include "../../common/database/PVDBQuery.h"
#include "../../common/database/PVDBInfos.h"

#include <QDialog>
#include <QSqlQueryModel>

namespace PVRush {

class PVDBPreviewWidget: public QDialog, public Ui::DbPreview
{
public:
	PVDBPreviewWidget(PVDBInfos const& infos, QString const& query, uint32_t nrows, QDialog* parent = 0);
public:
	bool init();
	void preview();

protected:
	bool _init;
	PVDBServ_p _serv;

	PVDBInfos _infos;
	QString _query_str;
	uint32_t _nrows;

protected:
	QSqlQueryModel* _table_model;
};

}

#endif
