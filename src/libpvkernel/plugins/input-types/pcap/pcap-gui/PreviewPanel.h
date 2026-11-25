/* * MIT License
 *
 * © Squey, 2026
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

#ifndef PCAP_GUI_PREVIEWPANEL_H
#define PCAP_GUI_PREVIEWPANEL_H

#include "Background.h"

#include <libpvpcap/profile.h>

#include <QAbstractTableModel>
#include <QStringList>
#include <QWidget>

#include <map>
#include <string>
#include <vector>

class QComboBox;
class QLabel;
class QStackedWidget;
class QTableView;
class QTimer;

namespace PVPcapsicum
{

//! What the import makes of the first frames of a capture.
struct preview_rows {
	//! The frame and the conversation of each row, then the columns of the profile.
	QStringList columns;
	std::vector<QStringList> rows;
	//! Of each column of the profile: how many rows have a value.
	std::vector<size_t> filled;
	//! Read of the capture, which may be all of them.
	uint64_t frames = 0;
	//! What stopped the extraction, empty if nothing did.
	QString error;
};

class PreviewModel : public QAbstractTableModel
{
	Q_OBJECT

  public:
	using QAbstractTableModel::QAbstractTableModel;

	void set(preview_rows rows);
	const preview_rows& rows() const { return _rows; }

	int rowCount(const QModelIndex& parent = {}) const override;
	int columnCount(const QModelIndex& parent = {}) const override;
	QVariant data(const QModelIndex& index, int role) const override;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

  private:
	preview_rows _rows;
};

/**
 * What the import makes of the first frames of one of the captures, extracted as the import
 * extracts them, again once the profile stops changing.
 */
class PreviewPanel : public QWidget
{
	Q_OBJECT

  public:
	//! Of a capture, read for a preview: a few tenths of a second.
	static constexpr uint64_t FRAMES = 1000;

	explicit PreviewPanel(QWidget* parent = nullptr);

	//! Previews the profile shortly, unless it changes again meanwhile.
	void preview(const QStringList& captures, const pvpcap::profile& p);

  Q_SIGNALS:
	/**
	 * A preview was shown: of `frames` frames read of `capture`, `rows` were kept, and so
	 * many of them had a value in each column, from 0 to 1.
	 */
	void previewed(const std::map<std::string, double>& filled,
	               size_t rows,
	               uint64_t frames,
	               const QString& capture);

  private:
	void run();
	void show_message(const QString& message, const QColor& color);

	QComboBox* _capture;
	QLabel* _status;
	QStackedWidget* _stack;
	QTableView* _table;
	QLabel* _message;
	PreviewModel* _model;
	QTimer* _waiting;
	BackgroundJob* _job;
	QStringList _captures;
	pvpcap::profile _profile;
	QString _previewed_capture;
};

} // namespace PVPcapsicum

#endif // PCAP_GUI_PREVIEWPANEL_H
