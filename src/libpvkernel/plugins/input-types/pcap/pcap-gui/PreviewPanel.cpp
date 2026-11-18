//
// MIT License
//
// © Squey, 2026
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

#include "PreviewPanel.h"

#include "Style.h"

#include <libpvpcap.h>
#include <libpvpcap/extraction.h>
#include <libpvpcap/ws.h>

#include <pvkernel/rush/PVNrawCacheManager.h>

#include <QComboBox>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QScrollBar>
#include <QStackedWidget>
#include <QTableView>
#include <QTimer>
#include <QVBoxLayout>

#include <chrono>
#include <thread>

namespace PVPcapsicum
{

namespace
{

//! The frame and the conversation, which the import shows before the columns of the profile.
constexpr int OWN_COLUMNS = 2;

/**
 * The values of a line of the extraction, as the import reads them: tshark escapes the
 * separator and the control characters with a backslash.
 */
QStringList values(std::string_view fields)
{
	QStringList values;
	std::string value;
	for (size_t i = 0; i < fields.size(); i++) {
		const char c = fields[i];
		if (c == '\\' and i + 1 < fields.size()) {
			const char escaped = fields[++i];
			value += escaped == 't' or escaped == 'n' or escaped == 'r' ? ' ' : escaped;
		} else if (c == pvpcap::SEPARATOR[0]) {
			values << QString::fromStdString(value);
			value.clear();
		} else {
			value += c;
		}
	}
	values << QString::fromStdString(value);
	return values;
}

} // namespace

void PreviewModel::set(preview_rows rows)
{
	beginResetModel();
	_rows = std::move(rows);
	endResetModel();
}

int PreviewModel::rowCount(const QModelIndex& parent) const
{
	return parent.isValid() ? 0 : int(_rows.rows.size());
}

int PreviewModel::columnCount(const QModelIndex& parent) const
{
	return parent.isValid() ? 0 : int(_rows.columns.size());
}

QVariant PreviewModel::data(const QModelIndex& index, int role) const
{
	if (not index.isValid()) {
		return {};
	}
	const QStringList& row = _rows.rows[index.row()];
	switch (role) {
	case Qt::DisplayRole:
	case Qt::ToolTipRole:
		return index.column() < row.size() ? row[index.column()] : QString();
	case Qt::ForegroundRole:
		return index.column() < OWN_COLUMNS ? QVariant(style::secondary()) : QVariant();
	case Qt::TextAlignmentRole:
		return index.column() < OWN_COLUMNS ? QVariant(Qt::AlignRight | Qt::AlignVCenter)
		                                    : QVariant();
	default:
		return {};
	}
}

QVariant PreviewModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (orientation != Qt::Horizontal or section >= _rows.columns.size()) {
		return {};
	}
	const int column = section - OWN_COLUMNS;
	const bool empty = column >= 0 and not _rows.rows.empty() and _rows.filled[column] == 0;
	switch (role) {
	case Qt::DisplayRole:
		return _rows.columns[section];
	case Qt::ToolTipRole:
		if (column < 0) {
			return section == 0 ? tr("The number of the frame, which Squey adds")
			                    : tr("The conversation of the frame, which Squey adds");
		}
		return empty ? tr("No row of the preview has a value for %1.").arg(_rows.columns[section])
		             : tr("%1 of the %2 rows of the preview have a value.")
		                   .arg(_rows.filled[column])
		                   .arg(_rows.rows.size());
	case Qt::ForegroundRole:
		return empty ? QVariant(style::warning()) : QVariant();
	case Qt::FontRole: {
		if (not empty) {
			return {};
		}
		QFont font;
		font.setItalic(true);
		return font;
	}
	default:
		return {};
	}
}

PreviewPanel::PreviewPanel(QWidget* parent)
    : QWidget(parent)
    , _model(new PreviewModel(this))
    , _waiting(new QTimer(this))
    , _job(new BackgroundJob(this))
{
	setAttribute(Qt::WA_StyledBackground);
	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(6);

	auto* header = new QHBoxLayout;
	header->addWidget(style::section_title(tr("Preview")));
	_status = new QLabel;
	_status->setObjectName("preview_status");
	style::color_label(_status, style::secondary(), true);
	header->addWidget(_status, 1);
	_capture = new QComboBox;
	_capture->setObjectName("preview_capture");
	_capture->setToolTip(tr("The capture to preview"));
	_capture->setSizeAdjustPolicy(QComboBox::AdjustToContents);
	_capture->hide();
	header->addWidget(_capture);
	layout->addLayout(header);

	_stack = new QStackedWidget;
	_table = new QTableView;
	_table->setObjectName("preview_table");
	_table->setModel(_model);
	_table->verticalHeader()->hide();
	_table->verticalHeader()->setDefaultSectionSize(_table->fontMetrics().height() + 6);
	_table->horizontalHeader()->setHighlightSections(false);
	_table->setSelectionMode(QAbstractItemView::NoSelection);
	_table->setWordWrap(false);
	_table->setTextElideMode(Qt::ElideRight);
	_stack->addWidget(_table);
	_message = new QLabel;
	_message->setObjectName("preview_message");
	_message->setAlignment(Qt::AlignCenter);
	_message->setWordWrap(true);
	_message->setTextInteractionFlags(Qt::TextSelectableByMouse);
	_stack->addWidget(_message);
	layout->addWidget(_stack, 1);

	_waiting->setSingleShot(true);
	_waiting->setInterval(350);
	connect(_waiting, &QTimer::timeout, this, &PreviewPanel::run);
	connect(_capture, &QComboBox::currentIndexChanged, this, [this]() { _waiting->start(0); });
	show_message(tr("Add captures to preview what they import."), style::secondary());
}

void PreviewPanel::preview(const QStringList& captures, const pvpcap::profile& p)
{
	if (captures != _captures) {
		const QString chosen = _capture->currentData().toString();
		const QSignalBlocker blocker(_capture);
		_capture->clear();
		for (const QString& capture : captures) {
			_capture->addItem(QFileInfo(capture).fileName(), capture);
		}
		_capture->setCurrentIndex(std::max(0, int(captures.indexOf(chosen))));
		_capture->setVisible(captures.size() > 1);
		_captures = captures;
	} else if (p == _profile and _job->running()) {
		return;
	}
	_profile = p;
	_waiting->start();
}

void PreviewPanel::run()
{
	_job->cancel();
	const QString capture = _capture->currentData().toString();
	if (capture.isEmpty()) {
		show_message(tr("Add captures to preview what they import."), style::secondary());
		_status->clear();
		return;
	}
	if (_profile.columns.empty()) {
		show_message(tr("Choose columns to preview them."), style::secondary());
		_status->clear();
		return;
	}
	_status->setText(tr("Reading the first frames of %1…").arg(QFileInfo(capture).fileName()));

	const pvpcap::profile p = _profile;
	const std::string work = PVRush::PVNrawCacheManager::nraw_dir().toStdString();
	_job->run(
	    [capture, p, work](const std::atomic<bool>& canceled) {
		    preview_rows result;
		    result.columns << tr("frame") << tr("stream");
		    for (const pvpcap::profile_field& column : p.columns) {
			    result.columns << QString::fromStdString(column.name);
		    }
		    result.filled.assign(p.columns.size(), 0);
		    try {
			    pvpcap::extraction::options opts = pvpcap::ws_extraction_options(p);
			    opts.work_directory = work;
			    opts.workers = 1;
			    opts.most_frames = FRAMES;
			    pvpcap::extraction extraction(capture.toStdString(), opts);
			    // next() waits for tshark, which a cancellation stops from another thread.
			    struct watcher {
				    std::atomic<bool> done = false;
				    std::thread thread;
				    ~watcher()
				    {
					    done = true;
					    thread.join();
				    }
			    } watching;
			    watching.thread = std::thread([&]() {
				    while (not watching.done and not canceled) {
					    std::this_thread::sleep_for(std::chrono::milliseconds(20));
				    }
				    if (canceled) {
					    extraction.cancel();
				    }
			    });
			    pvpcap::extraction::line line;
			    while (extraction.next(line)) {
				    if (canceled) {
					    return result;
				    }
				    QStringList row;
				    row << QString::number(line.frame)
				        << (line.stream == pvpcap::NO_STREAM ? QString()
				                                             : QString::number(line.stream));
				    const QStringList fields = values(line.fields);
				    for (qsizetype i = 0; i < fields.size() and size_t(i) < p.columns.size(); i++) {
					    result.filled[i] += not fields[i].isEmpty();
				    }
				    row << fields;
				    result.rows.push_back(std::move(row));
			    }
			    result.frames = extraction.frames();
		    } catch (const std::exception& e) {
			    result.error = QString::fromStdString(e.what());
		    }
		    return result;
	    },
	    [this, capture, p](preview_rows result) {
		    _previewed_capture = capture;
		    const QString name = QFileInfo(capture).fileName();
		    if (not result.error.isEmpty()) {
			    _status->setText(tr("tshark could not read %1.").arg(name));
			    show_message(result.error, style::error());
			    Q_EMIT previewed({}, 0, 0, capture);
			    return;
		    }
		    std::map<std::string, double> filled;
		    size_t empty = 0;
		    if (not result.rows.empty()) {
			    for (size_t i = 0; i < p.columns.size(); i++) {
				    filled[p.columns[i].name] = double(result.filled[i]) / result.rows.size();
				    empty += result.filled[i] == 0;
			    }
		    }
		    const size_t rows = result.rows.size();
		    const uint64_t frames = result.frames;
		    const bool all = frames < FRAMES;
		    QString status =
		        all ? tr("All the %1 frames of %2").arg(style::count_text(frames)).arg(name)
		            : tr("The first %1 frames of %2").arg(style::count_text(frames)).arg(name);
		    if (rows != frames) {
			    status += tr(", of which the filter keeps %1").arg(style::count_text(rows));
		    }
		    if (empty != 0) {
			    status += " · " + (empty == 1 ? tr("1 column is empty")
			                                  : tr("%1 columns are empty").arg(empty));
		    }
		    _status->setText(status);
		    // Where the table was scrolled to, if it shows the same columns again.
		    const bool same = result.columns == _model->rows().columns;
		    const int scrolled = _table->horizontalScrollBar()->value();
		    _model->set(std::move(result));
		    if (rows == 0) {
			    show_message(tr("The filter keeps none of these frames."), style::warning());
		    } else {
			    _stack->setCurrentWidget(_table);
			    _table->resizeColumnsToContents();
			    for (int c = 0; c < _model->columnCount(); c++) {
				    _table->setColumnWidth(c, std::min(_table->columnWidth(c), 320));
			    }
			    if (same) {
				    _table->horizontalScrollBar()->setValue(scrolled);
			    }
		    }
		    Q_EMIT previewed(filled, rows, frames, capture);
	    });
}

void PreviewPanel::show_message(const QString& message, const QColor& color)
{
	style::color_label(_message, color);
	_message->setText(message);
	_stack->setCurrentWidget(_message);
}

} // namespace PVPcapsicum
