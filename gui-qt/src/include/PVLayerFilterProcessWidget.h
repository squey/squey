#ifndef PVINSPECTOR_PVLAYERFILTERPROCESSWIDGET_H
#define PVINSPECTOR_PVLAYERFILTERPROCESSWIDGET_H

#include <picviz/PVLayerFilter.h>
#include "PVArgumentListWidget.h"
#include "PVTabSplitter.h"

namespace PVInspector {

class PVLayerFilterProcessWidget: public PVArgumentListWidget
{
	Q_OBJECT

public:
	PVLayerFilterProcessWidget(PVTabSplitter* tab, PVFilter::PVArgumentList& args, Picviz::PVLayerFilter_p filter_p);
	virtual ~PVLayerFilterProcessWidget();

public slots:
	void save_Slot();
	void preview_Slot();
	void cancel_Slot();

protected:
	virtual void create_btns();
	virtual void set_btns_layout();
	virtual void connect_btns();

private:
	static void process_layer_filter(Picviz::PVLayerFilter* filter, Picviz::PVLayer* layer);

protected:
	PVTabSplitter* _tab;
	Picviz::PVView_p _view;
	Picviz::PVLayerFilter_p _filter_p;
	QPushButton* _help_btn;
	QPushButton* _preview_btn;
	Picviz::PVLayer _pre_filter_layer_org;
	PVFilter::PVArgumentList _args_org;

private:
	bool _has_changed;
};

}

#endif
