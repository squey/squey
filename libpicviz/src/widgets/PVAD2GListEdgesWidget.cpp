#include <picviz/widgets/PVAD2GEdgeEditor.h>
#include <picviz/widgets/PVAD2GListEdgesWidget.h>
#include <picviz/widgets/PVAD2GFunctionPropertiesWidget.h>

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
	add_edge_list_f(QTableWidget* table, Picviz::PVView const* cur_va, Picviz::PVView const* cur_vb, bool& edge_found):
		_table(table),
		_cur_idx(0),
		_cur_va(cur_va),
		_cur_vb(cur_vb),
		_cur_edge_found(&edge_found)
	{
		edge_found = false;
	}

public:
	void operator()(Picviz::PVCombiningFunctionView& cf, Picviz::PVView& va, Picviz::PVView& vb) const
	{
		size_t idx_row = _cur_idx;

		if (&va == _cur_va && &vb == _cur_vb) {
			*_cur_edge_found = true;
		}

		QTableWidgetItem* item = new QTableWidgetItem(QString::number(va.get_display_view_id()));
		item->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
		QVariant var;
		var.setValue<void*>(&va);
		item->setData(Qt::UserRole, var);
		var.setValue<void*>(&cf);
		item->setData(Qt::UserRole+1, var);
		_table->setItem(idx_row, 0, item);

		item = new QTableWidgetItem(QString::number(vb.get_display_view_id()));
		item->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
		var.setValue<void*>(&vb);
		item->setData(Qt::UserRole, var);
		_table->setItem(idx_row, 1, item);
		_cur_idx++;
	}

private:
	mutable QTableWidget* _table;
	mutable size_t _cur_idx;
	mutable QWidget* _parent;
	Picviz::PVView const* _cur_va;
	Picviz::PVView const* _cur_vb;
	bool* _cur_edge_found;
};

}

PVWidgets::PVAD2GListEdgesWidget::PVAD2GListEdgesWidget(Picviz::PVAD2GView& graph, QWidget* parent):
	QWidget(parent),
	_graph(graph),
	_cur_cf(NULL)
{

	_edges_table = new QTableWidget(this);
	_edges_table->setColumnCount(2);
	_edges_table->verticalHeader()->hide();
	_edges_table->setHorizontalHeaderLabels(QStringList() << tr("Source view") << tr("Destination view"));
	_edges_table->setSelectionBehavior(QAbstractItemView::SelectRows);
	_edges_table->setSelectionMode(QAbstractItemView::SingleSelection);

	_edge_properties_widget = new PVWidgets::PVAD2GEdgeEditor(this);
	_edge_properties_widget->hide();
	_function_properties_widget = new PVAD2GFunctionPropertiesWidget(/*_view_org, _view_dst, *rff,*/ this);
	_function_properties_widget->hide();

	// Connection
	connect(_edges_table, SIGNAL(cellDoubleClicked(int, int)), this, SLOT(show_edge(int, int)));
	connect(_function_properties_widget, SIGNAL(function_properties_changed(const Picviz::PVSelRowFilteringFunction_p &)), this, SLOT(update_edge_editor_Slot(const Picviz::PVSelRowFilteringFunction_p &)));
	connect(_edge_properties_widget, SIGNAL(update_fonction_properties(const Picviz::PVView&, const Picviz::PVView&, Picviz::PVSelRowFilteringFunction_p& )), this, SLOT(update_fonction_properties(const Picviz::PVView &, const Picviz::PVView &, Picviz::PVSelRowFilteringFunction_p &)));
	connect(_edge_properties_widget, SIGNAL(cur_rff_removed()), _function_properties_widget, SLOT(hide()));

	QHBoxLayout* main_layout = new QHBoxLayout();
	main_layout->addWidget(_edges_table);
	main_layout->addWidget(_edge_properties_widget);
	main_layout->addWidget(_function_properties_widget);

	setLayout(main_layout);
	setFocusPolicy(Qt::StrongFocus);
}

void PVWidgets::PVAD2GListEdgesWidget::show_edge(int row, int /*column*/)
{
	// Get views and CF
	QTableWidgetItem* item_src = _edges_table->item(row, 0);
	QTableWidgetItem* item_dst = _edges_table->item(row, 1);

	Picviz::PVView* view_src = (Picviz::PVView*) item_src->data(Qt::UserRole).value<void*>();
	Picviz::PVView* view_dst = (Picviz::PVView*) item_dst->data(Qt::UserRole).value<void*>();
	_cur_cf = (Picviz::PVCombiningFunctionView*) item_src->data(Qt::UserRole+1).value<void*>();

	_edge_properties_widget->set_cf(*view_src, *view_dst, *_cur_cf);

	_edge_properties_widget->show();
}

void PVWidgets::PVAD2GListEdgesWidget::update_edge_editor_Slot(const Picviz::PVSelRowFilteringFunction_p & rff)
{
	_edge_properties_widget->update_item_Slot(rff);
}

void PVWidgets::PVAD2GListEdgesWidget::update_fonction_properties(const Picviz::PVView& src_view, const Picviz::PVView& dst_view, Picviz::PVSelRowFilteringFunction_p& rff)
{
	_function_properties_widget->set_views(src_view, dst_view);
	_function_properties_widget->set_current_rff(rff.get());
	_function_properties_widget->show();
}

void PVWidgets::PVAD2GListEdgesWidget::update_list_edges()
{
	bool edge_found = false;
	__impl::add_edge_list_f f(_edges_table, _edge_properties_widget->get_view_org(), _edge_properties_widget->get_view_dst(), edge_found);
	_edges_table->setRowCount(_graph.get_edges_count());
	_graph.visit_edges(f);
	if (!edge_found) {
		_edge_properties_widget->hide();
		_function_properties_widget->hide();
	}
}
