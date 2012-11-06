/**
 * \file PVCombiningFunctionView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <picviz/PVCombiningFunctionView.h>
#include <picviz/PVTFViewRowFiltering.h>
#include <picviz/PVView.h>

Picviz::PVCombiningFunctionView::PVCombiningFunctionView()
{
	// AG: for now, the only TF in here is a row filtering one
	PVTransformationFunctionView_p tf_p(new PVTFViewRowFiltering());
	_tfs.push_back(tf_p);
}

void Picviz::PVCombiningFunctionView::pre_process(PVView const& view_src, PVView const& view_dst)
{
	foreach(PVTransformationFunctionView_p const& tf, _tfs) {
		tf->pre_process(view_src, view_dst);
	}
}

Picviz::PVSelection Picviz::PVCombiningFunctionView::operator()(PVView const& view_src, PVView const& view_dst) const
{
	// AG: this is now hard-coded in here, the idea is to have the user being able to modify this in a close future...
	Picviz::PVSelection const& sel_src = view_src.get_real_output_selection();
	if (_tfs.size() == 0) {
		return sel_src;
	}

	std::vector<PVSelection> out_sel;
	out_sel.reserve(_tfs.size());
	foreach(PVTransformationFunctionView_p const& tf, _tfs) {
		out_sel.emplace_back(std::move((*tf)(view_src, view_dst, sel_src)));
	}

	// Merge with an OR operation
	// For instance, the user could choose the operation he wants to do here !
	Picviz::PVSelection& ret(out_sel.front()); 
	std::vector<PVSelection>::const_iterator it_sel = out_sel.begin();
	it_sel++;
	for (; it_sel != out_sel.end(); it_sel++) {
		ret.or_optimized(*it_sel);
	}

	return std::move(ret);
}

Picviz::PVTFViewRowFiltering* Picviz::PVCombiningFunctionView::get_first_tf()
{
	return dynamic_cast<PVTFViewRowFiltering*>(_tfs[0].get());
}

Picviz::PVTFViewRowFiltering const* Picviz::PVCombiningFunctionView::get_first_tf() const
{
	return dynamic_cast<PVTFViewRowFiltering const*>(_tfs[0].get());
}

void Picviz::PVCombiningFunctionView::from_string(std::string const& str)
{
	QDomDocument xml_doc;
	if (!xml_doc.setContent(QByteArray(str.c_str(), str.size()), false)) {
		return;
	}

	QDomElement elt_tf = xml_doc.firstChildElement("tf");
	if (elt_tf.isNull()) {
		return;
	}
	get_first_tf()->from_xml(elt_tf);
}

void Picviz::PVCombiningFunctionView::to_string(std::string& str) const
{
	// AG: for now, this is a hand-made QDomDocument.
	// What we will have to do in the future (:/) :
	//  * have a clear interface for PVSerializeArchive, and move to
	//    a different class the directory-specific properties and methods
	//  * make a PVSerializeArchiveXml backend
	//  * use this backend to create/read the xml file
	
	QDomDocument xml_doc;
	QDomElement first_tf = xml_doc.createElement("tf");
	get_first_tf()->to_xml(first_tf);
	xml_doc.appendChild(first_tf);

	str = xml_doc.toByteArray(4).constData();
}
