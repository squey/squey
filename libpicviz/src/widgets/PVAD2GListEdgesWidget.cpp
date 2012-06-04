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
	add_edge_list_f(QTableWidget* table, Picviz::PVView const* cur_va, Picviz::PVView const* cur_vb, int& edge_index):
		_table(table),
		_cur_idx(0),
		_cur_va(cur_va),
		_cur_vb(cur_vb),
		_cur_edge_index(&edge_index)
	{
		edge_index = -1;
	}

public:
	void operator()(Picviz::PVCombiningFunctionView& cf, Picviz::PVView& va, Picviz::PVView& vb) const
	{
		size_t idx_row = _cur_idx;

		if (&va == _cur_va && &vb == _cur_vb) {
			*_cur_edge_index = _cur_idx;
		}

		QTableWidgetItem* item = new QTableWidgetItem(QString::number(va.get_display_view_id()));
		item->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
		item->setTextAlignment(Qt::AlignCenter);
		QVariant var;
		var.setValue<void*>(&va);
		item->setData(Qt::UserRole, var);
		var.setValue<void*>(&cf);
		item->setData(Qt::UserRole+1, var);
		_table->setItem(idx_row, 0, item);

		item = new QTableWidgetItem(QString::number(vb.get_display_view_id()));
		item->setTextAlignment(Qt::AlignCenter);
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
	int* _cur_edge_index;
};

class PVTableWidgetTest: public QTableWidget
{
public:
	PVTableWidgetTest(QWidget* parent=NULL):
		QTableWidget(parent)
	{ }

public:
	QSize sizeHint() const
	{
		return QSize(horizontalHeader()->size().width() + 20, 40);
	}
};

}

PVWidgets::PVAD2GListEdgesWidget::PVAD2GListEdgesWidget(Picviz::PVAD2GView& graph, QWidget* parent):
	QWidget(parent),
	_graph(graph),
	_cur_cf(NULL)
{

	_edges_table = (QTableWidget*) (new __impl::PVTableWidgetTest(this));
	_edges_table->setColumnCount(2);
	_edges_table->verticalHeader()->hide();
	_edges_table->setHorizontalHeaderItem(0, new QTableWidgetItem(tr("From")));
	_edges_table->setHorizontalHeaderItem(1, new QTableWidgetItem(tr("To")));
	_edges_table->setSelectionBehavior(QAbstractItemView::SelectRows);
	_edges_table->setSelectionMode(QAbstractItemView::SingleSelection);
	_edges_table->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::MinimumExpanding);
	_edges_table->resizeColumnsToContents();
	_edges_table->horizontalHeader()->setStretchLastSection(true);
	_edges_table->horizontalHeader()->setResizeMode(QHeaderView::Fixed);
	_edges_table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

	_removeAct = new QAction(QIcon(), tr("Remove"), this);
	//_edges_table->addAction(_removeAct);
	_edges_table->setContextMenuPolicy(Qt::ActionsContextMenu);
	connect(_removeAct, SIGNAL(triggered()), this, SLOT(remove_Slot()));
	_removeAct->setEnabled(false);

	_edge_properties_widget = new PVWidgets::PVAD2GEdgeEditor(this);
	_edge_properties_widget->setEnabled(false);
	_function_properties_widget = new PVAD2GFunctionPropertiesWidget(/*_view_org, _view_dst, *rff,*/ this);
	_function_properties_widget->hide();

	// Connection
	connect(_edges_table, SIGNAL(currentItemChanged(QTableWidgetItem*, QTableWidgetItem*)), this, SLOT(selection_changed_Slot(QTableWidgetItem*, QTableWidgetItem*)));
	connect(_function_properties_widget, SIGNAL(function_properties_changed(const Picviz::PVSelRowFilteringFunction_p &)), this, SLOT(update_edge_editor_Slot(const Picviz::PVSelRowFilteringFunction_p &)));
	connect(_edge_properties_widget, SIGNAL(update_fonction_properties(const Picviz::PVView&, const Picviz::PVView&, Picviz::PVSelRowFilteringFunction_p& )), this, SLOT(update_fonction_properties(const Picviz::PVView &, const Picviz::PVView &, Picviz::PVSelRowFilteringFunction_p &)));
	connect(_edge_properties_widget, SIGNAL(cur_rff_removed()), _function_properties_widget, SLOT(hide()));

	QHBoxLayout* main_layout = new QHBoxLayout();
	main_layout->addWidget(_edges_table);
	main_layout->addWidget(_edge_properties_widget);
	main_layout->addWidget(_function_properties_widget);
	main_layout->addStretch(1);

	setLayout(main_layout);
	setFocusPolicy(Qt::StrongFocus);
}

void PVWidgets::PVAD2GListEdgesWidget::select_row(int src_view_id, int dst_view_id)
{
	for (int i=0; i<_edges_table->rowCount(); i++) {
		if (_edges_table->item(i, 0)->text() == QString::number(src_view_id) && _edges_table->item(i, 1)->text() == QString::number(dst_view_id)) {
			_edges_table->selectRow(i);
			show_edge(i);
			break;
		}
	}
}

void PVWidgets::PVAD2GListEdgesWidget::clear_current_edge()
{
	_function_properties_widget->hide();
	_edges_table->clearSelection();
	_edge_properties_widget->setEnabled(false);
	_edge_properties_widget->set_no_cf();
}

void PVWidgets::PVAD2GListEdgesWidget::selection_changed_Slot(QTableWidgetItem* cur, QTableWidgetItem* prev)
{
	if (cur && cur != prev) {
		show_edge(cur->row());
	}
}

void PVWidgets::PVAD2GListEdgesWidget::remove_Slot()
{
	int row = _edges_table->currentIndex().row();
	QTableWidgetItem* item_src = _edges_table->item(row, 0);
	QTableWidgetItem* item_dst = _edges_table->item(row, 1);
	Picviz::PVView* view_src = (Picviz::PVView*) item_src->data(Qt::UserRole).value<void*>();
	Picviz::PVView* view_dst = (Picviz::PVView*) item_dst->data(Qt::UserRole).value<void*>();

	_edges_table->removeRow(row);
	_graph.del_edge_f(view_src, view_dst);
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

	_graph.set_selected_edge(view_src, view_dst);

	_edge_properties_widget->setEnabled(true);
}

void PVWidgets::PVAD2GListEdgesWidget::update_edge_editor_Slot(const Picviz::PVSelRowFilteringFunction_p & rff)
{
	_edge_properties_widget->update_item_Slot(rff);
}

void PVWidgets::PVAD2GListEdgesWidget::update_fonction_properties(const Picviz::PVView& src_view, const Picviz::PVView& dst_view, Picviz::PVSelRowFilteringFunction_p& rff)
{
	_function_properties_widget->set_views(src_view, dst_view);
	_function_properties_widget->set_current_rff(rff.get(), false);
	_function_properties_widget->show();
}

void PVWidgets::PVAD2GListEdgesWidget::update_list_edges()
{
	int edge_new_index = -1;
	__impl::add_edge_list_f f(_edges_table, _edge_properties_widget->get_view_org(), _edge_properties_widget->get_view_dst(), edge_new_index);
	size_t nedges =  _graph.get_edges_count();
	_removeAct->setEnabled(nedges);

	// unselecting any edge in the graph view
	_graph.set_selected_edge(0, 0);

	// clearing the edge table
	for (int i = 0; i < _edges_table->rowCount(); ++i) {
		_edges_table->removeRow(0);
	}

	// and filling it with valid edges
	_edges_table->setRowCount(nedges);
	_graph.visit_edges(f);

	if ((edge_new_index != -1) && _edges_table->isEnabled()) {
		_edges_table->selectRow(edge_new_index);
		return;
	}

	if (! nedges) {
		_edge_properties_widget->setEnabled(false);
		_function_properties_widget->hide();
	}
}
