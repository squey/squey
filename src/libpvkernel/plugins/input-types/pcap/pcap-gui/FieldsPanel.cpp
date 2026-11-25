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

#include "FieldsPanel.h"

#include "Style.h"

#include <libpvpcap/ws.h>

#include <pvkernel/widgets/PVImportWidget.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QHeaderView>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMimeData>
#include <QPainter>
#include <QStyledItemDelegate>
#include <QTimer>
#include <QTreeView>
#include <QVBoxLayout>

#include <algorithm>
#include <unordered_map>

namespace PVPcapsicum
{

namespace
{

//! Of a search: more would only be scrolled through.
constexpr size_t MOST_FOUND = 200;

QString percent_text(double share)
{
	if (share <= 0) {
		return {};
	}
	if (share < 0.01) {
		return "<1%";
	}
	return QString::number(int(share * 100 + 0.5)) + "%";
}

//! Draws the share of the frames holding a protocol or a field as a bar, then as a number.
class ShareDelegate : public QStyledItemDelegate
{
  public:
	using QStyledItemDelegate::QStyledItemDelegate;

	void paint(QPainter* painter,
	           const QStyleOptionViewItem& option,
	           const QModelIndex& index) const override
	{
		QStyleOptionViewItem background = option;
		initStyleOption(&background, index);
		background.text.clear();
		option.widget->style()->drawControl(QStyle::CE_ItemViewItem, &background, painter,
		                                    option.widget);
		const double share = index.data(FieldTreeModel::SHARE_ROLE).toDouble();
		if (share < 0) {
			return;
		}
		painter->save();
		const QRect area = option.rect.adjusted(6, 4, -6, -4);
		const QRect bar(area.left(), area.center().y() - 2, int(area.width() * 0.4), 4);
		QColor track = style::secondary();
		track.setAlphaF(0.25);
		painter->setPen(Qt::NoPen);
		painter->setBrush(track);
		painter->drawRoundedRect(bar, 2, 2);
		painter->setBrush(style::accent());
		painter->drawRoundedRect(QRect(bar.left(), bar.top(),
		                               std::max(2, int(bar.width() * std::min(share, 1.0))),
		                               bar.height()),
		                         2, 2);
		painter->setPen(option.palette.color(
		    option.state & QStyle::State_Selected ? QPalette::HighlightedText : QPalette::Text));
		painter->drawText(option.rect.adjusted(6, 0, -6, 0), Qt::AlignRight | Qt::AlignVCenter,
		                  percent_text(share));
		painter->restore();
	}
};

} // namespace

sampled_fields merge_samples(const std::vector<const pvpcap::capture_sample*>& samples)
{
	sampled_fields merged;
	std::unordered_map<std::string, size_t> protocols;
	for (const pvpcap::capture_sample* sample : samples) {
		merged.frames += sample->frames;
		for (const auto& protocol : sample->protocols) {
			const auto [it, added] = protocols.try_emplace(protocol.name, merged.protocols.size());
			if (added) {
				merged.protocols.push_back(protocol);
			} else {
				auto& known = merged.protocols[it->second];
				known.frames += protocol.frames;
				known.depth = std::min(known.depth, protocol.depth);
			}
		}
		for (const auto& [name, field] : sample->fields) {
			auto& known = merged.fields[name];
			known.frames += field.frames;
			if (known.example.empty()) {
				known.example = field.example;
			}
		}
	}
	std::stable_sort(merged.protocols.begin(), merged.protocols.end(),
	                 [](const auto& a, const auto& b) {
		                 return a.depth != b.depth ? a.depth < b.depth : a.frames > b.frames;
	                 });
	return merged;
}

FieldTreeModel::FieldTreeModel(QObject* parent) : QAbstractItemModel(parent) {}

void FieldTreeModel::set_catalog(std::shared_ptr<const pvpcap::field_catalog> catalog)
{
	_catalog = std::move(catalog);
	rebuild();
}

void FieldTreeModel::set_samples(sampled_fields samples)
{
	_samples = std::move(samples);
	rebuild();
}

void FieldTreeModel::set_columns(const std::vector<pvpcap::profile_field>& columns)
{
	if (columns == _columns) {
		return;
	}
	_columns = columns;
	_column_names.clear();
	for (const pvpcap::profile_field& column : columns) {
		_column_names.insert(column.name);
	}
	for (node& n : _nodes) {
		if (n.type == node::kind::protocol) {
			n.columns = 0;
			for (const uint32_t field : _catalog->protocols()[n.index].fields) {
				n.columns += is_column(field);
			}
		}
	}
	const auto changed = [this](const QModelIndex& parent, size_t count) {
		if (count != 0) {
			Q_EMIT dataChanged(index(0, NAME, parent), index(int(count) - 1, EXAMPLE, parent));
		}
	};
	changed({}, _roots.size());
	for (size_t i = 0; i < _nodes.size(); i++) {
		changed(createIndex(_nodes[i].row, 0, quintptr(i)), _nodes[i].children.size());
	}
}

size_t FieldTreeModel::set_search(const QString& search, size_t most)
{
	_search = search;
	_most = most;
	rebuild();
	return _found;
}

int FieldTreeModel::add(node n, int parent)
{
	n.parent = parent;
	const int index = int(_nodes.size());
	_nodes.push_back(std::move(n));
	std::vector<int>& siblings = parent < 0 ? _roots : _nodes[parent].children;
	_nodes[index].row = int(siblings.size());
	siblings.push_back(index);
	return index;
}

void FieldTreeModel::add_fields(int protocol, const std::vector<uint32_t>& fields, bool split)
{
	const auto frames_of = [this](uint32_t field) -> uint64_t {
		const auto it = _samples.fields.find(std::string(_catalog->fields()[field].filter_name));
		return it != _samples.fields.end() ? it->second.frames : 0;
	};
	// First what the captures hold, or else the columns: the others below, folded.
	std::vector<std::pair<uint64_t, uint32_t>> seen;
	std::vector<uint32_t> others;
	for (const uint32_t field : fields) {
		const uint64_t frames = frames_of(field);
		if (split ? frames != 0 : is_column(field)) {
			seen.emplace_back(frames, field);
		} else {
			others.push_back(field);
		}
	}
	std::stable_sort(seen.begin(), seen.end(),
	                 [](const auto& a, const auto& b) { return a.first > b.first; });
	for (const auto& [frames, field] : seen) {
		add({node::kind::field, field, -1, 0, {}, frames, 0}, protocol);
	}
	if (not others.empty()) {
		const int more =
		    add({node::kind::others, uint32_t(others.size()), -1, 0, {}, 0, 0}, protocol);
		for (const uint32_t field : others) {
			add({node::kind::field, field, -1, 0, {}, 0, 0}, more);
		}
	}
}

void FieldTreeModel::rebuild()
{
	beginResetModel();
	_nodes.clear();
	_roots.clear();
	_found = 0;
	if (_catalog and not _search.trimmed().isEmpty()) {
		const auto bonus = [this](uint32_t field) {
			const auto it =
			    _samples.fields.find(std::string(_catalog->fields()[field].filter_name));
			return it == _samples.fields.end() ? 0
			                                   : 600 + int(400 * _samples.share(it->second.frames));
		};
		for (const uint32_t field :
		     _catalog->search(_search.toStdString(), _most, bonus, &_found)) {
			const auto it =
			    _samples.fields.find(std::string(_catalog->fields()[field].filter_name));
			add({node::kind::field,
			     field,
			     -1,
			     0,
			     {},
			     it != _samples.fields.end() ? it->second.frames : 0,
			     0},
			    -1);
		}
	} else if (_catalog) {
		std::vector<std::pair<std::string, uint64_t>> protocols = {{"_ws.col", _samples.frames}};
		const auto listed = [&](const std::string& name) {
			return std::any_of(protocols.begin(), protocols.end(),
			                   [&](const auto& p) { return p.first == name; });
		};
		if (_samples.frames != 0) {
			for (const auto& protocol : _samples.protocols) {
				if (not listed(protocol.name)) {
					protocols.emplace_back(protocol.name, protocol.frames);
				}
			}
		} else {
			protocols.emplace_back("frame", 0);
			for (const pvpcap::profile_field& column : _columns) {
				const int64_t field = _catalog->find_field(column.name);
				if (field >= 0) {
					const std::string name(
					    _catalog->protocols()[_catalog->fields()[field].protocol].filter_name);
					if (not listed(name)) {
						protocols.emplace_back(name, 0);
					}
				}
			}
		}
		for (const auto& [name, frames] : protocols) {
			const int64_t protocol = _catalog->find_protocol(name);
			if (protocol < 0 or _catalog->protocols()[protocol].fields.empty()) {
				continue;
			}
			const int n = add({node::kind::protocol, uint32_t(protocol), -1, 0, {}, frames, 0}, -1);
			const auto& fields = _catalog->protocols()[protocol].fields;
			if (name == "_ws.col") {
				// Every frame has them, which tshark does not write as fields of the frames.
				for (const uint32_t field : fields) {
					add({node::kind::field, field, -1, 0, {}, _samples.frames, 0}, n);
				}
			} else {
				add_fields(n, fields, _samples.frames != 0);
			}
			for (const uint32_t field : fields) {
				_nodes[n].columns += is_column(field);
			}
		}
	}
	endResetModel();
}

bool FieldTreeModel::is_column(uint32_t field) const
{
	return _column_names.contains(std::string(_catalog->fields()[field].filter_name));
}

bool FieldTreeModel::importable(uint32_t field) const
{
	return not pvpcap::ws_disabled_fields.contains(
	    std::string(_catalog->fields()[field].filter_name));
}

QModelIndex FieldTreeModel::index(int row, int column, const QModelIndex& parent) const
{
	const std::vector<int>& siblings =
	    parent.isValid() ? _nodes[parent.internalId()].children : _roots;
	if (row < 0 or row >= int(siblings.size()) or column < 0 or column >= COLUMN_COUNT) {
		return {};
	}
	return createIndex(row, column, quintptr(siblings[row]));
}

QModelIndex FieldTreeModel::parent(const QModelIndex& index) const
{
	if (not index.isValid()) {
		return {};
	}
	const int parent = _nodes[index.internalId()].parent;
	return parent < 0 ? QModelIndex() : createIndex(_nodes[parent].row, 0, quintptr(parent));
}

int FieldTreeModel::rowCount(const QModelIndex& parent) const
{
	if (parent.column() > 0) {
		return 0;
	}
	return int(parent.isValid() ? _nodes[parent.internalId()].children.size() : _roots.size());
}

int FieldTreeModel::columnCount(const QModelIndex&) const
{
	return COLUMN_COUNT;
}

QVariant FieldTreeModel::data(const QModelIndex& index, int role) const
{
	if (not index.isValid()) {
		return {};
	}
	const node& n = _nodes[index.internalId()];
	const int column = index.column();

	if (role == SHARE_ROLE) {
		return n.type == node::kind::others or n.frames == 0 ? -1.0 : _samples.share(n.frames);
	}
	if (role == Qt::ToolTipRole) {
		return tooltip(n);
	}

	if (n.type == node::kind::protocol) {
		const auto& protocol = _catalog->protocols()[n.index];
		switch (role) {
		case Qt::DisplayRole:
			if (column == NAME) {
				return style::to_qstring(protocol.short_name);
			}
			if (column == DESCRIPTION) {
				return style::to_qstring(protocol.name);
			}
			if (column == EXAMPLE and n.columns != 0) {
				return n.columns == 1 ? tr("1 column") : tr("%1 columns").arg(n.columns);
			}
			return {};
		case Qt::FontRole: {
			QFont font;
			font.setBold(true);
			return font;
		}
		case Qt::ForegroundRole:
			return column == EXAMPLE ? QVariant(style::accent()) : QVariant();
		default:
			return {};
		}
	}

	if (n.type == node::kind::others) {
		switch (role) {
		case Qt::DisplayRole:
			if (column == NAME) {
				return n.index == 1 ? tr("1 more field") : tr("%1 more fields").arg(n.index);
			}
			return column == DESCRIPTION and _samples.frames != 0
			           ? tr("none of the first frames holds them")
			           : QVariant();
		case Qt::FontRole: {
			QFont font;
			font.setItalic(true);
			return font;
		}
		case Qt::ForegroundRole:
			return QVariant(style::secondary());
		default:
			return {};
		}
	}

	const auto& field = _catalog->fields()[n.index];
	switch (role) {
	case Qt::DisplayRole:
		switch (column) {
		case NAME:
			return style::to_qstring(field.filter_name);
		case DESCRIPTION:
			return style::to_qstring(field.name);
		case EXAMPLE: {
			const auto it = _samples.fields.find(std::string(field.filter_name));
			return it != _samples.fields.end() ? QString::fromStdString(it->second.example)
			                                   : QString();
		}
		default:
			return {};
		}
	case Qt::CheckStateRole:
		if (column == NAME) {
			return is_column(n.index) ? Qt::Checked : Qt::Unchecked;
		}
		return {};
	case FIELD_ROLE:
		return style::to_qstring(field.filter_name);
	case Qt::ForegroundRole:
		return column == EXAMPLE or column == DESCRIPTION ? QVariant(style::secondary())
		                                                  : QVariant();
	default:
		return {};
	}
}

QString FieldTreeModel::tooltip(const node& n) const
{
	if (n.type == node::kind::others) {
		return tr("The fields of this protocol that none of the first frames of the captures "
		          "holds: later frames may.");
	}
	if (n.type == node::kind::protocol) {
		const auto& protocol = _catalog->protocols()[n.index];
		QString text = QString("<b>%1</b> — %2")
		                   .arg(style::to_qstring(protocol.filter_name).toHtmlEscaped(),
		                        style::to_qstring(protocol.name).toHtmlEscaped());
		if (n.frames != 0) {
			text += "<br>" + tr("In %1 of the first frames of the captures.")
			                     .arg(percent_text(_samples.share(n.frames)));
		}
		return text;
	}
	const auto& field = _catalog->fields()[n.index];
	const std::string name(field.filter_name);
	QString text = QString("<b>%1</b> — %2")
	                   .arg(QString::fromStdString(name).toHtmlEscaped(),
	                        style::to_qstring(field.name).toHtmlEscaped());
	if (not field.description.empty()) {
		text += "<br>" + style::to_qstring(field.description).toHtmlEscaped();
	}
	text +=
	    "<br>" +
	    tr("%1, imported as %2").arg(style::to_qstring(field.type), style::type_text(field.type));
	const auto seen = _samples.fields.find(name);
	if (seen != _samples.fields.end()) {
		text += "<br>" + tr("In %1 of the first frames of the captures, such as “%2”.")
		                     .arg(percent_text(_samples.share(seen->second.frames)),
		                          QString::fromStdString(seen->second.example).toHtmlEscaped());
	}
	if (pvpcap::ws_needs_two_passes(name, std::string(field.type))) {
		text += "<br><i>" +
		        tr("tshark only writes it right when it reads the captures twice, which takes "
		           "about a quarter longer.") +
		        "</i>";
	}
	if (not importable(n.index)) {
		text += "<br><i>" +
		        tr("It can't be imported: Squey numbers the frames and the conversations itself, "
		           "and the values of this field would be wrong or cut.") +
		        "</i>";
	}
	return text;
}

bool FieldTreeModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
	if (not index.isValid() or role != Qt::CheckStateRole) {
		return false;
	}
	const node& n = _nodes[index.internalId()];
	if (n.type != node::kind::field or not importable(n.index)) {
		return false;
	}
	const auto& field = _catalog->fields()[n.index];
	Q_EMIT toggled({std::string(field.filter_name), std::string(field.type)},
	               value.toInt() == Qt::Checked);
	return true;
}

QVariant FieldTreeModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (orientation != Qt::Horizontal or role != Qt::DisplayRole) {
		return {};
	}
	switch (section) {
	case NAME:
		return tr("Field");
	case DESCRIPTION:
		return tr("Name");
	case SHARE:
		return tr("Frames");
	case EXAMPLE:
		return tr("Example");
	default:
		return {};
	}
}

Qt::ItemFlags FieldTreeModel::flags(const QModelIndex& index) const
{
	if (not index.isValid()) {
		return Qt::NoItemFlags;
	}
	const node& n = _nodes[index.internalId()];
	if (n.type != node::kind::field) {
		return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
	}
	if (not importable(n.index)) {
		return Qt::ItemIsSelectable;
	}
	return Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable |
	       Qt::ItemIsDragEnabled;
}

QStringList FieldTreeModel::mimeTypes() const
{
	return {MIME_TYPE};
}

QMimeData* FieldTreeModel::mimeData(const QModelIndexList& indexes) const
{
	QStringList fields;
	for (const QModelIndex& index : indexes) {
		const QString field = index.data(FIELD_ROLE).toString();
		if (index.column() == NAME and not field.isEmpty()) {
			fields << field;
		}
	}
	auto* mime = new QMimeData;
	mime->setData(MIME_TYPE, fields.join('\n').toUtf8());
	return mime;
}

FieldsPanel::FieldsPanel(QWidget* parent)
    : QWidget(parent), _model(new FieldTreeModel(this)), _searching(new QTimer(this))
{
	setAttribute(Qt::WA_StyledBackground);
	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(6);
	layout->addWidget(PVWidgets::PVImportWidget::section_title(tr("Fields")));

	_search = new QLineEdit;
	_search->setObjectName("fields_search");
	_search->setClearButtonEnabled(true);
	_search->addAction(PVModdedIcon("magnifying-glass"), QLineEdit::LeadingPosition);
	_search->setPlaceholderText(tr("Search the fields of tshark: user agent, dns.qry… (Ctrl+F)"));
	_search->installEventFilter(this);
	layout->addWidget(_search);

	_view = new QTreeView;
	_view->setObjectName("fields_tree");
	_view->setModel(_model);
	_view->setUniformRowHeights(true);
	_view->setSelectionMode(QAbstractItemView::ExtendedSelection);
	_view->setDragEnabled(true);
	_view->setDragDropMode(QAbstractItemView::DragOnly);
	_view->setItemDelegateForColumn(FieldTreeModel::SHARE, new ShareDelegate(_view));
	_view->installEventFilter(this);
	_view->header()->setStretchLastSection(true);
	_view->header()->resizeSection(FieldTreeModel::NAME, 220);
	_view->header()->resizeSection(FieldTreeModel::DESCRIPTION, 170);
	_view->header()->resizeSection(FieldTreeModel::SHARE, 84);
	layout->addWidget(_view, 1);

	_status = new QLabel;
	_status->setObjectName("fields_status");
	_status->setWordWrap(true);
	style::color_label(_status, style::secondary(), true);
	layout->addWidget(_status);

	_searching->setSingleShot(true);
	_searching->setInterval(150);
	connect(_searching, &QTimer::timeout, this, &FieldsPanel::search);
	connect(_search, &QLineEdit::textChanged, _searching, qOverload<>(&QTimer::start));
	connect(_model, &FieldTreeModel::toggled, this, &FieldsPanel::toggled);
	connect(_view, &QTreeView::doubleClicked, this, &FieldsPanel::toggle);
	update_status();
}

void FieldsPanel::set_catalog(std::shared_ptr<const pvpcap::field_catalog> catalog,
                              const QString& problem)
{
	_catalog_loaded = catalog != nullptr;
	_catalog_problem = problem;
	if (catalog) {
		_search->setPlaceholderText(
		    tr("Search the %1 fields of tshark: user agent, dns.qry… (Ctrl+F)")
		        .arg(style::count_text(catalog->fields().size())));
	}
	_model->set_catalog(std::move(catalog));
	search();
}

void FieldsPanel::set_samples(sampled_fields samples, bool reading)
{
	_reading = reading;
	_model->set_samples(std::move(samples));
	search();
}

void FieldsPanel::set_columns(const std::vector<pvpcap::profile_field>& columns)
{
	_model->set_columns(columns);
}

void FieldsPanel::focus_search()
{
	_search->setFocus();
	_search->selectAll();
}

void FieldsPanel::reveal(const std::string& name)
{
	_search->setText(QString::fromStdString(name));
	search();
	if (_model->rowCount() != 0) {
		const QModelIndex first = _model->index(0, FieldTreeModel::NAME);
		_view->setCurrentIndex(first);
		_view->scrollTo(first);
	}
}

void FieldsPanel::search()
{
	_searching->stop();
	_found = _model->set_search(_search->text(), MOST_FOUND);
	if (_search->text().trimmed().isEmpty()) {
		// The protocols holding columns, to see them among the others.
		for (int row = 0; row < _model->rowCount(); row++) {
			const QModelIndex protocol = _model->index(row, FieldTreeModel::NAME);
			if (not _model->index(row, FieldTreeModel::EXAMPLE).data().toString().isEmpty()) {
				_view->expand(protocol);
			}
		}
	}
	update_status();
}

void FieldsPanel::update_status()
{
	QString text;
	if (not _catalog_problem.isEmpty()) {
		style::color_label(_status, style::error(), true);
		_status->setText(_catalog_problem);
		return;
	}
	if (not _catalog_loaded) {
		text = tr("Reading the fields tshark knows…");
	} else if (not _search->text().trimmed().isEmpty()) {
		text = _found == 0            ? tr("No field holds these words: try fewer or shorter ones.")
		       : _found <= MOST_FOUND ? tr("%1 fields found.").arg(style::count_text(_found))
		                              : tr("The %1 likeliest of %2 fields found.")
		                                    .arg(MOST_FOUND)
		                                    .arg(style::count_text(_found));
	} else if (_reading and _model->samples().frames == 0) {
		text = tr("Reading the first frames of the captures…");
	} else if (_model->samples().frames != 0) {
		text = tr("The protocols and the fields of the first %1 frames of the captures, the most "
		          "frequent first.")
		           .arg(style::count_text(_model->samples().frames));
	} else {
		text = tr("Add captures to see the protocols and the fields they hold, or search the "
		          "fields of tshark.");
	}
	style::color_label(_status, style::secondary(), true);
	_status->setText(text);
}

void FieldsPanel::toggle(const QModelIndex& index)
{
	const QModelIndex check = index.siblingAtColumn(FieldTreeModel::NAME);
	if (check.flags() & Qt::ItemIsUserCheckable) {
		_model->setData(check,
		                check.data(Qt::CheckStateRole).toInt() == Qt::Checked ? Qt::Unchecked
		                                                                      : Qt::Checked,
		                Qt::CheckStateRole);
	}
}

bool FieldsPanel::eventFilter(QObject* watched, QEvent* event)
{
	if (watched == _view and event->type() == QEvent::KeyPress) {
		const auto* key = static_cast<QKeyEvent*>(event);
		if (key->key() == Qt::Key_Return or key->key() == Qt::Key_Enter) {
			// Rather than importing, which the dialog does on Enter.
			toggle(_view->currentIndex());
			return true;
		}
	}
	if (watched == _search and event->type() == QEvent::KeyPress) {
		const auto* key = static_cast<QKeyEvent*>(event);
		if (key->key() == Qt::Key_Down and _model->rowCount() != 0) {
			_view->setFocus();
			if (not _view->currentIndex().isValid()) {
				_view->setCurrentIndex(_model->index(0, FieldTreeModel::NAME));
			}
			return true;
		}
		if (key->key() == Qt::Key_Return or key->key() == Qt::Key_Enter) {
			// Rather than importing, which the dialog does on Enter: checks the first field.
			search();
			QModelIndex first = _model->index(0, FieldTreeModel::NAME);
			while (first.isValid() and not(first.flags() & Qt::ItemIsUserCheckable)) {
				first = _model->index(0, FieldTreeModel::NAME, first);
			}
			if (first.isValid() and first.data(Qt::CheckStateRole).toInt() != Qt::Checked) {
				_model->setData(first, Qt::Checked, Qt::CheckStateRole);
			}
			return true;
		}
	}
	return QWidget::eventFilter(watched, event);
}

} // namespace PVPcapsicum
