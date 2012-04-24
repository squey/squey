#include <picviz/widgets/PVAD2GListEdgesWidget.h>

#include <picviz/PVAD2GView.h>
#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVView.h>

#include <QHeaderView>
#include <QVBoxLayout>
#include <QObject>

namespace __impl {

struct add_edge_list_f
{
public:
	add_edge_list_f(QTableWidget* table, QWidget* parent /* =0*/):
		_table(table),
		_cur_idx(0),
		_parent(parent)
	{ }

public:
	void operator()(Picviz::PVCombiningFunctionView& cf, Picviz::PVView& va, Picviz::PVView& vb) const
	{
		PVWidgets::PVAD2GEdgeEditor* edge_widget = new PVWidgets::PVAD2GEdgeEditor(va, vb, cf);
		QObject::connect(edge_widget, SIGNAL(update_fonction_properties(const Picviz::PVView&, const Picviz::PVView&, Picviz::PVSelRowFilteringFunction_p& )), _parent, SLOT(update_fonction_properties(const Picviz::PVView &, const Picviz::PVView &, Picviz::PVSelRowFilteringFunction_p &)));
		size_t idx_row = _cur_idx;
		_table->setItem(idx_row, 0, new QTableWidgetItem(QString::number(va.get_view_id()+1)));
		_table->setItem(idx_row, 1, new QTableWidgetItem(QString::number(vb.get_view_id()+1)));
		_table->setCellWidget(idx_row, 2, edge_widget);
		//_table->setRowHeight(idx_row, edge_widget->height());
		_table->setRowHeight(idx_row, 150);
		_table->setColumnWidth(2, picviz_max(_table->columnWidth(2), edge_widget->width()));
		_cur_idx++;
	}

private:
	mutable QTableWidget* _table;
	mutable size_t _cur_idx;
	mutable QWidget* _parent;
};

}

PVWidgets::PVAD2GListEdgesWidget::PVAD2GListEdgesWidget(Picviz::PVAD2GView& graph, QWidget* parent):
	QWidget(parent),
	_graph(graph),
	_current_edge_widget(NULL)
{

	_edges_table = new QTableWidget(this);
	_edges_table->setColumnCount(3);
	_edges_table->verticalHeader()->hide();
	_edges_table->setHorizontalHeaderLabels(QStringList() << tr("Source view") << tr("Destination view") << tr("Link functions"));
	_edges_table->setSelectionBehavior(QAbstractItemView::SelectRows);
	_edges_table->setSelectionMode(QAbstractItemView::SingleSelection);

	_function_properties_widget = new PVAD2GFunctionPropertiesWidget(/*_view_org, _view_dst, *rff,*/ this);

	// Connection
	connect(_function_properties_widget, SIGNAL(function_properties_changed(const Picviz::PVSelRowFilteringFunction_p &)), this, SLOT(update_edge_editor_Slot(const Picviz::PVSelRowFilteringFunction_p &)));

	QHBoxLayout* main_layout = new QHBoxLayout();
	main_layout->addWidget(_edges_table);
	main_layout->addWidget(_function_properties_widget);

	setLayout(main_layout);
	setFocusPolicy(Qt::StrongFocus);
}

void PVWidgets::PVAD2GListEdgesWidget::update_edge_editor_Slot(const Picviz::PVSelRowFilteringFunction_p & rff)
{
	if(_current_edge_widget) {
		_current_edge_widget->update_item_Slot(rff);
	}
}

void PVWidgets::PVAD2GListEdgesWidget::update_fonction_properties(const Picviz::PVView& src_view, const Picviz::PVView& dst_view, Picviz::PVSelRowFilteringFunction_p& rff)
{
	_current_edge_widget = (PVAD2GEdgeEditor*) sender();
	_function_properties_widget->set_views(src_view, dst_view);
	_function_properties_widget->set_current_rff(rff.get());
}

void PVWidgets::PVAD2GListEdgesWidget::update_list_edges()
{
	_edges_table->setRowCount(_graph.get_edges_count());
	_graph.visit_edges(__impl::add_edge_list_f(_edges_table, this));
	//_edges_table->resizeColumnsToContents();

}
