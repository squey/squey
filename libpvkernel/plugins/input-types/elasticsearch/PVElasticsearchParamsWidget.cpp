/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include "PVElasticsearchParamsWidget.h"
#include "PVInputTypeElasticsearch.h"

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/widgets/PVFilterableComboBox.h>

#include <QMessageBox>
#include <QPushButton>

static const char* query_types[] = {"Query Builder", "JSON", "SQL"};
static constexpr const char MATCH_ALL_QUERY[] = R"###({ "query" : { "match_all" : { } } })###";

PVRush::PVElasticsearchParamsWidget::PVElasticsearchParamsWidget(
    PVInputTypeElasticsearch const* in_t, PVRush::hash_formats const& formats, QWidget* parent)
    : PVParamsWidget<PVInputTypeElasticsearch,
                     PVElasticsearchPresets,
                     PVElasticsearchInfos,
                     PVElasticsearchQuery>(in_t, formats, parent)
{
	QLabel* label_index = new QLabel("Index :");
	_btn_refresh = new QPushButton("&Refresh");
	_combo_index = new PVWidgets::PVFilterableComboBox();
	_port_sb->setValue(PVElasticsearchAPI::DEFAULT_PORT);

	_btn_refresh->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Minimum);
	_combo_index->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Minimum);

	QHBoxLayout* custom_layout = new QHBoxLayout();
	custom_layout->addWidget(label_index);
	custom_layout->addWidget(_combo_index);
	custom_layout->addWidget(_btn_refresh);

	_custom_layout->addLayout(custom_layout);

	_columns_tree_widget->header()->setSectionResizeMode(QHeaderView::ResizeToContents);

	connect(_combo_index,
	        static_cast<void (QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged), this,
	        &PVElasticsearchParamsWidget::index_changed_slot);
	connect(_combo_index, static_cast<void (QComboBox::*)(int)>(&QComboBox::activated), this,
	        &PVElasticsearchParamsWidget::index_changed_by_user_slot);
	connect(_btn_refresh, &QPushButton::clicked, this,
	        &PVElasticsearchParamsWidget::fetch_server_data_slot);

	for (const char* const qtype_name : query_types) {
		_query_type_cb->addItem(qtype_name);
	}

	_help_label->setText(
	    "<html>"
	    "<head/>"
	    "<body>"
	    "<p><span style=\" text-decoration: underline;\">Elasticsearch limitation:</span><br/></p>"
	    "<p><ul><li>The order of the lines returned by Elasticsearch queries does not respect the "
	    "order of the lines as imported by Logstash.</li>"
	    "<li>Requests use tokenized strings. If filters were applied during import, the search "
	    "have to respect their constraints.<br/>"
	    "<i>Eg : If a lowercase filter was used at import time, the according search must be done "
	    "in lowercase too.</i></li></ul>"
	    "</p><br/>"
	    "<p><span style=\" text-decoration: underline;\">Logstash CSV plugin "
	    "limitation:</span><br/></p>"
	    "<p>When importing logs into Elasticsearch using Logstash CSV plugin, be careful not to "
	    "name any of your columns "
	    "<span style=\" font-weight:600;\">message</span>, <span style=\" "
	    "font-weight:600;\">type</span>, "
	    "<span style=\" font-weight:600;\">host</span>,<span style=\" font-weight:600;\"> "
	    "path</span> or "
	    "<span style=\" font-weight:600;\">geoip </span>because these are internally used by "
	    "Logstash and will likely to cause conflicts.</p>"
	    "</body>"
	    "</html>");
}

QString PVRush::PVElasticsearchParamsWidget::get_sql_query_prefix() const
{
	return QString("SELECT * FROM %1 WHERE ").arg(get_infos().get_index());
}

void PVRush::PVElasticsearchParamsWidget::index_changed_slot(const QString& index)
{
	buttonBox->buttons()[0]->setEnabled(index != "");

	QString query_type = _query_type_cb->currentText();

	if (query_type == query_types[EQueryType::SQL]) {
		_gb_query->setTitle(get_sql_query_prefix() + " ...");
	}
}

void PVRush::PVElasticsearchParamsWidget::reset_columns_tree_widget()
{
	disconnect(_columns_tree_widget, &QTreeWidget::itemChanged, this,
	           &PVElasticsearchParamsWidget::tree_item_changed);

	_columns_tree_widget->clear();
	_root_item = new QTreeWidgetItem(_columns_tree_widget);
	_root_item->setExpanded(true);
	_root_item->setText(0, "properties");
	_root_item->setCheckState(0, Qt::Unchecked);
	std::vector<QTreeWidgetItem*> parents({_root_item});

	PVRush::PVElasticsearchAPI es(get_infos());
	es.visit_columns([&](const std::string& rel_name, const std::string& abs_name,
	                     const std::string& type, bool is_leaf, bool is_last_child) {
		QTreeWidgetItem* tree_item = new QTreeWidgetItem(parents.back());

		tree_item->setText(0, rel_name.c_str());
		tree_item->setData(0, Qt::UserRole, QString::fromStdString(abs_name));
		tree_item->setText(1, type.c_str());
		tree_item->setExpanded(true);
		tree_item->setCheckState(0, Qt::Unchecked);

		if (not is_leaf) { // node
			parents.emplace_back(tree_item);
		}

		if (is_last_child) {
			parents.pop_back();
		}
	});

	_columns_tree_widget->resizeColumnToContents(0);

	connect(_columns_tree_widget, &QTreeWidget::itemChanged, this,
	        &PVElasticsearchParamsWidget::tree_item_changed);
}

/*
 * Recursively visit all the columns of the columns tree widget
 *
 * @param tree_item root item to visit
 * @param f the function that is called against each QTreeWidgetItem
 */
static void visit_columns(QTreeWidgetItem* tree_item,
                          const std::function<void(QTreeWidgetItem*)>& f)
{
	f(tree_item);
	for (int i = 0; i < tree_item->childCount(); i++) {
		QTreeWidgetItem* tree_child = tree_item->child(i);
		visit_columns(tree_child, f);
	}
}

/*
 * Recursively visit the checked columns of the columns tree widget
 *
 * @param tree_item root item to visit
 * @param f the function that is called against each QTreeWidgetItem
 */
static void visit_selected_columns(QTreeWidgetItem* tree_item,
                                   const std::function<void(QTreeWidgetItem*)>& f)
{
	static auto is_checked = [](const QTreeWidgetItem* item) {
		return item->checkState(0) == Qt::Checked or item->checkState(0) == Qt::PartiallyChecked;
	};

	if (is_checked(tree_item) and tree_item->childCount() == 0) {
		f(tree_item);
	}

	for (int i = 0; i < tree_item->childCount(); i++) {
		QTreeWidgetItem* tree_child = tree_item->child(i);
		if (is_checked(tree_child)) {
			visit_selected_columns(tree_child, f);
		}
	}
}

static void propagate_check_state(QTreeWidgetItem* leaf_item)
{
	QTreeWidgetItem* parent = leaf_item->parent();
	if (parent == nullptr) {
		return;
	}

	int child_count = parent->childCount();
	int checked_count = 0;
	for (int i = 0; i < child_count; i++) {
		checked_count += parent->child(i)->checkState(0);
	}
	Qt::CheckState parent_state =
	    (checked_count == child_count * 2
	         ? Qt::Checked
	         : (checked_count == 0 ? Qt::Unchecked : Qt::PartiallyChecked));

	parent->setCheckState(0, parent_state);
	propagate_check_state(parent);
}

void PVRush::PVElasticsearchParamsWidget::tree_item_changed(QTreeWidgetItem* item, int column)
{
	_columns_tree_widget->blockSignals(true);

	if (column == 0) {
		if (item->childCount() > 0) { // node
			Qt::CheckState check_state = item->checkState(0);
			visit_columns(item,
			              [&](QTreeWidgetItem* child) { child->setCheckState(0, check_state); });
		}
		propagate_check_state(item);
	}

	_columns_tree_widget->blockSignals(false);
}

void PVRush::PVElasticsearchParamsWidget::index_changed_by_user_slot()
{
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::QUERY_BUILDER) {
		PVRush::PVElasticsearchAPI es(get_infos());
		_querybuilder->set_filters(es.querybuilder_columns());
	}

	reset_columns_tree_widget();
}

void PVRush::PVElasticsearchParamsWidget::fetch_server_data_slot()
{
	fetch_server_data(get_infos());
}

bool PVRush::PVElasticsearchParamsWidget::check_connection(std::string* error /*= nullptr*/)
{
	const PVElasticsearchInfos& infos = get_infos();

	PVRush::PVElasticsearchAPI es(infos);

	bool ret = es.check_connection(error);

	if (ret) {
		fetch_server_data(infos);
	}

	return ret;
}

void PVRush::PVElasticsearchParamsWidget::query_type_changed_slot()
{
	int query_type = _query_type_cb->currentIndex();

	_querybuilder->setVisible(false);
	_txt_query->setVisible(false);

	if (query_type == EQueryType::SQL) {
		_txt_query->setPlainText("");
		_reference_label->setText(
		    "<a href=\"https://github.com/NLPchina/elasticsearch-sql/\">"
		    "<span style=\" text-decoration: underline; color:#0000ff;\">Elasticsearch SQL plugin");
		PVRush::PVElasticsearchAPI es(get_infos());
		if (es.is_sql_available()) {
			_gb_query->setTitle(get_sql_query_prefix() + " ...");
			_txt_query->setEnabled(true);
		} else {
			_txt_query->setPlainText("Please, install the SQL plugin to your Elasticsearch "
			                         "instance to support this feature.");
			_txt_query->setEnabled(false);
			// buttonBox->buttons()[0]->setEnabled(false);
		}
		_txt_query->setVisible(true);
	} else if (query_type == EQueryType::QUERY_BUILDER) {
		_gb_query->setTitle("Query");
		_reference_label->setText("");
		_querybuilder->reset_rules();

		PVRush::PVElasticsearchAPI es(get_infos());
		_querybuilder->set_filters(es.querybuilder_columns());

		_querybuilder->setVisible(true);
	} else { // EQueryType::JSON
		_txt_query->setPlainText(MATCH_ALL_QUERY);
		_gb_query->setTitle("Query");
		_reference_label->setText("<a "
		                          "href=\"https://www.elastic.co/guide/en/elasticsearch/reference/"
		                          "current/query-dsl-filters.html\">"
		                          "<span style=\" text-decoration: underline; "
		                          "color:#0000ff;\">Elasticsearch Filters reference");
		_txt_query->setVisible(true);
		_txt_query->setEnabled(true);
	}
}

/*****************************************************************************
 * PVRush::PVElasticsearchParamsWidget::get_export_filters
 *****************************************************************************/

QString PVRush::PVElasticsearchParamsWidget::get_export_filters()
{
	return "CSV File (*.csv)";
}

void PVRush::PVElasticsearchParamsWidget::export_query_result(QTextStream& output_stream,
                                                              PVCore::PVProgressBox& pbox,
                                                              std::string* error /*= nullptr*/)
{
	size_t count = 0;
	bool query_end = false;

	PVRush::PVElasticsearchAPI es(get_infos());
	const PVElasticsearchQuery& query = get_query(error);

	pbox.set_maximum(es.count(query));

	do {
		PVElasticsearchAPI::rows_t rows;
		query_end = not es.extract(query, rows, error);

		if (error && error->empty() == false) {
			return;
		}

		if (pbox.get_cancel_state() == PVCore::PVProgressBox::CancelState::CANCEL ||
		    pbox.get_cancel_state() == PVCore::PVProgressBox::CancelState::CANCEL2) {
			break;
		}

		for (const std::vector<std::string>& row : rows) {
			std::string v;
			for (const std::string& col : row) {
				v += PVRush::PVUtils::safe_export(col, ",", "\"") + ",";
			}
			v.resize(v.size() - 1); // remove last separator
			output_stream << v.c_str() << "\n";
		}
		count += rows.size();

		if (output_stream.status() == QTextStream::WriteFailed) {
			if (error) {
				*error = "Write failed. Is your disk full ?";
			}
		}

		pbox.set_value(count);
		pbox.set_extended_status(QString("%L1 rows exported so far").arg(count));
	} while (query_end == false);
}

bool PVRush::PVElasticsearchParamsWidget::set_infos(PVElasticsearchInfos const& infos)
{
	bool res = PVParamsWidget<PVInputTypeElasticsearch, PVElasticsearchPresets,
	                          PVElasticsearchInfos, PVElasticsearchQuery>::set_infos(infos);
	_combo_index->setCurrentIndex(_combo_index->findText(infos.get_index()));

	if (infos.get_index().isEmpty()) {
		return true;
	}

	// update filter_path
	reset_columns_tree_widget();
	const std::string& filter_path = infos.get_filter_path().toStdString();
	std::unordered_set<std::string> selected_columns;
	boost::split(selected_columns, filter_path, boost::is_any_of(","));

	visit_columns(_root_item, [&](QTreeWidgetItem* item) {
		const std::string& abs_name = item->data(0, Qt::UserRole).toString().toStdString();
		if (selected_columns.find(abs_name) != selected_columns.end()) {
			item->setCheckState(0, Qt::Checked);
		}
	});

	return res;
}

PVRush::PVElasticsearchInfos PVRush::PVElasticsearchParamsWidget::get_infos() const
{
	PVRush::PVElasticsearchInfos infos =
	    PVParamsWidget<PVInputTypeElasticsearch, PVElasticsearchPresets, PVElasticsearchInfos,
	                   PVElasticsearchQuery>::get_infos();

	infos.set_index(_combo_index->currentText());
	infos.set_filter_path(_format_path->text());

	// update filter_path
	if (_root_item) {
		std::vector<std::string> selected_columns;
		visit_selected_columns(_root_item, [&](const QTreeWidgetItem* item) {
			selected_columns.emplace_back(item->data(0, Qt::UserRole).toString().toStdString());
		});
		const std::string& filter_path = boost::algorithm::join(selected_columns, ",");
		infos.set_filter_path(QString::fromStdString(filter_path));
	}

	return infos;
}

QString
PVRush::PVElasticsearchParamsWidget::get_server_query(std::string* error /* = nullptr */) const
{
	QString q = get_serialize_query();
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::JSON) {
		q = _txt_query->toPlainText();
	} else {
		PVRush::PVElasticsearchAPI es(get_infos());

		if (query_type == EQueryType::SQL) {
			q = es.sql_to_json(QString(get_sql_query_prefix() + q).toStdString(), error).c_str();
		} else if (query_type == EQueryType::QUERY_BUILDER) {
			q = QString::fromStdString(es.rules_to_json(_querybuilder->get_rules()));
			if (q.isEmpty()) {
				q = MATCH_ALL_QUERY;
			}
		}
	}

	while (q.endsWith('\n'))
		q.chop(1);

	return q;
}

QString PVRush::PVElasticsearchParamsWidget::get_serialize_query() const
{
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::QUERY_BUILDER) {
		return QString(_querybuilder->get_rules().c_str());
	} else {
		return _txt_query->toPlainText();
	}
}

void PVRush::PVElasticsearchParamsWidget::set_query(QString const& query)
{
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::QUERY_BUILDER) {
		if (not get_infos().get_index().isEmpty()) {
			PVRush::PVElasticsearchAPI es(get_infos());
			_querybuilder->set_filters(es.querybuilder_columns());
			_querybuilder->set_rules(query.toStdString());
		}
	} else {
		_txt_query->setPlainText(query);
	}
}

size_t PVRush::PVElasticsearchParamsWidget::query_result_count(std::string* error /* = nullptr */)
{
	std::string err;
	PVRush::PVElasticsearchAPI es(get_infos());
	const PVElasticsearchQuery& query = get_query(&err);

	if (err.empty() == false) {
		if (error) {
			*error = err;
		}
		return 0;
	}

	return es.count(query, error);
}

bool PVRush::PVElasticsearchParamsWidget::fetch_server_data(const PVElasticsearchInfos& infos)
{
	PVRush::PVElasticsearchAPI es(infos);
	PVRush::PVElasticsearchAPI::indexes_t indexes = es.indexes();

	QString old_index = _combo_index->currentText();

	_combo_index->clear();
	QStringList indexes_list;
	for (const std::string& index : indexes) {
		indexes_list << (index.c_str());
	}
	_combo_index->set_string_list(indexes_list);

	_combo_index->setCurrentIndex(_combo_index->findText(old_index));

	return true;
}

size_t PVRush::PVElasticsearchParamsWidget::get_selected_columns_count() const
{
	size_t selected_columns_count = 0;

	visit_selected_columns(_root_item,
	                       [&](const QTreeWidgetItem* /*item*/) { selected_columns_count++; });

	return selected_columns_count;
}

static size_t get_first_level_fields_count(const QString& format_path)
{
	QDomDocument doc;

	QFile f(format_path);
	if (not f.open(QFile::ReadOnly | QFile::Text))
		return {};
	QTextStream in(&f);

	doc.setContent(in.readAll());

	const QDomElement& root = doc.documentElement();

	size_t fields_count = 0;
	for (QDomElement n = root.firstChildElement(); not n.isNull(); n = n.nextSiblingElement()) {
		fields_count += n.tagName() == PVFORMAT_XML_TAG_FIELD_STR;
	}

	return fields_count;
}

void PVRush::PVElasticsearchParamsWidget::accept()
{
	size_t fields_count = get_selected_columns_count();
	if (fields_count < 2) {
		QMessageBox::critical(this, "Invalid format error",
		                      "At least two columns must be selected to have a valid format.");
		return;
	}

	if (is_format_custom()) {
		update_custom_format();
	} else if (not(fields_count == get_first_level_fields_count(_format_path->text()))) {
		QMessageBox::critical(
		    this, "Invalid format error",
		    "Number of columns mismatchs between existing format and selected mapping.\n\n"
		    "You may want to edit and save your custom format.");
		return;
	}

	QDialog::accept();
}

void PVRush::PVElasticsearchParamsWidget::update_custom_format()
{
	_custom_format = PVElasticsearchAPI(get_infos()).get_format_from_mapping();
}

void PVRush::PVElasticsearchParamsWidget::edit_custom_format()
{
	update_custom_format();
	PVParamsWidgetBase::edit_custom_format();
}
