#include "include/OverviewWidget.h"
#include "ui_OverviewWidget.h"

#include <iostream>

OverviewWidget::OverviewWidget(const rapidjson::Document& json_data, QWidget* parent)
    : QWidget(parent), _ui(new Ui::OverviewWidget), _json_data(json_data)
{
	_ui->setupUi(this);

	// Set the number of columns in the tree
	_ui->tree_widget->setColumnCount(4);

	// Set the headers
	_ui->tree_widget->setHeaderLabels(QStringList() << "Name"
	                                                << "Filter"
	                                                << "Description"
	                                                << "Type");
}

OverviewWidget::~OverviewWidget()
{
	delete _ui;
}

void OverviewWidget::update_model()
{
	_ui->tree_widget->clear();

	fill_tree_widget(_json_data["children"]);

	_ui->tree_widget->expandAll();
	_ui->tree_widget->resizeColumnToContents(0);
}

void OverviewWidget::fill_tree_widget(const rapidjson::Value& value)
{
	for (auto& item : value.GetArray()) {
		rapidjson::Document selected_fields;
		selected_fields.SetArray();
		rapidjson::Document::AllocatorType& allocator = selected_fields.GetAllocator();

		for (auto& field : item["fields"].GetArray()) {
			if (field["select"].GetBool()) {
				rapidjson::Value val(field, allocator);
				selected_fields.GetArray().PushBack(val, allocator);
			}
		}

		if (not selected_fields.Empty()) {
			add_tree_root(item["name"].GetString(), item["filter_name"].GetString(),
			              item["short_name"].GetString(), selected_fields);
		}

		// recurse over children
		fill_tree_widget(item["children"]);
	}
}

void OverviewWidget::add_tree_root(QString name,
                                   QString filter,
                                   QString description,
                                   const rapidjson::Document& children)
{
	QList<QTreeWidgetItem*> root_list = _ui->tree_widget->findItems(filter, Qt::MatchExactly, 1);

	QTreeWidgetItem* root;
	if (root_list.isEmpty()) {
		root = new QTreeWidgetItem(_ui->tree_widget);
		root->setText(0, name);
		root->setText(1, filter);
		root->setText(2, description);
		root->setText(3, ""); // no type for root node
	} else {
		root = root_list[0];
	}

	for (auto& child : children.GetArray()) {
		add_tree_child(root, child["name"].GetString(), child["filter_name"].GetString(),
		               child["description"].GetString(), child["type"].GetString());
	}
}

void OverviewWidget::add_tree_child(
    QTreeWidgetItem* parent, QString name, QString filter, QString description, QString type)
{
	QTreeWidgetItem* child;

	// if the child is already inserted, do nothing
	for (int i = 0; i < parent->childCount(); i++) {
		child = parent->child(i);
		if (child->text(1) == filter) {
			return;
		}
	}

	// insert a new child
	child = new QTreeWidgetItem();

	child->setText(0, name);
	child->setText(1, filter);
	child->setText(2, description);
	child->setText(3, type);

	parent->addChild(child);
}