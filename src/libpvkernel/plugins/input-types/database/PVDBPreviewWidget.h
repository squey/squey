/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVDBPREVIEWWIDGET_FILE_H
#define PVDBPREVIEWWIDGET_FILE_H

#include <ui_db_preview.h>
#include "../../common/database/PVDBQuery.h"
#include "../../common/database/PVDBInfos.h"

#include <QDialog>
#include <QSqlQueryModel>

namespace PVRush
{

class PVDBPreviewWidget : public QDialog, public Ui::DbPreview
{
  public:
	PVDBPreviewWidget(PVDBInfos const& infos,
	                  QString const& query,
	                  uint32_t nrows,
	                  QDialog* parent = 0);

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
