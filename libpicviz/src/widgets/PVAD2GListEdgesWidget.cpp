#include <picviz/widgets/PVAD2GListEdgesWidget.h>
#include <picviz/widgets/PVAD2GEdgeEditor.h>

#include <picviz/PVAD2GView.h>
#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVView.h>

#include <QHeaderView>
#include <QVBoxLayout>

namespace __impl {

struct add_edge_list_f
{
	add_edge_list_f(QTableWidget* table):
		_table(table),
		_cur_idx(0)
	{ }

	void operator()(Picviz::PVCombiningFunctionView& cf, Picviz::PVView& va, Picviz::PVView& vb) const
	{
		PVWidgets::PVAD2GEdgeEditor* edge_widget = new PVWidgets::PVAD2GEdgeEditor(va, vb, cf);
		size_t idx_row = _cur_idx;
		_table->setItem(idx_row, 0, new QTableWidgetItem(va.get_window_name()));
		_table->setItem(idx_row, 1, new QTableWidgetItem(vb.get_window_name()));
		_table->setCellWidget(idx_row, 2, edge_widget);
		//_table->setRowHeight(idx_row, edge_widget->height());
		_table->setRowHeight(idx_row, 150);
		_table->setColumnWidth(2, picviz_max(_table->columnWidth(2), edge_widget->width()));
		_cur_idx++;
	}

private:
	mutable QTableWidget* _table;
	mutable size_t _cur_idx;
};

}

PVWidgets::PVAD2GListEdgesWidget::PVAD2GListEdgesWidget(Picviz::PVAD2GView& graph, QWidget* parent):
	QWidget(parent),
	_graph(graph)
{
	_edges_table = new QTableWidget(this);
	_edges_table->setColumnCount(3);
	_edges_table->verticalHeader()->hide();
	_edges_table->setHorizontalHeaderLabels(QStringList() << tr("Source view") << tr("Destination view") << tr("Link functions"));
	_edges_table->setSelectionBehavior(QAbstractItemView::SelectRows);
	_edges_table->setSelectionMode(QAbstractItemView::SingleSelection);

	QVBoxLayout* main_layout = new QVBoxLayout();
	main_layout->addWidget(_edges_table);

	setLayout(main_layout);
	setFocusPolicy(Qt::StrongFocus);
}

void PVWidgets::PVAD2GListEdgesWidget::update_list_edges()
{
	_edges_table->setRowCount(_graph.get_edges_count());
	_graph.visit_edges(__impl::add_edge_list_f(_edges_table));
}
