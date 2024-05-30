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
