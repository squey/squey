#include <pvkernel/core/PVClassLibrary.h>
#include <picviz/PVSelRowFilteringFunction.h>

void Picviz::PVSelRowFilteringFunction::to_xml(QDomElement& elt) const
{
	elt.setAttribute("plugin", registered_name());
	elt.setAttribute("op", (uint32_t) _combination_op);
	QDomElement args_elt = elt.ownerDocument().createElement("args");
	PVCore::PVArgumentList_to_QDomElement(get_args(), args_elt);
	elt.appendChild(args_elt);
}


Picviz::PVSelRowFilteringFunction_p Picviz::PVSelRowFilteringFunction::from_xml(QDomElement const& elt)
{
	QString plugin_name = elt.attribute("plugin", "");
	Picviz::PVSelRowFilteringFunction_p lib_rff = LIB_CLASS(Picviz::PVSelRowFilteringFunction)::get().get_class_by_name(plugin_name);

	QDomElement elt_args = elt.firstChildElement("args");
	uint32_t op = elt.attribute("op", QString()).toUInt(NULL);
	if (op > PVCore::PVBinaryOperation::LAST_BINOP) {
		op = PVCore::PVBinaryOperation::FIRST_BINOP;
	}

	if (!lib_rff || elt_args.isNull()) {
		return Picviz::PVSelRowFilteringFunction_p();
	}

	// Clone that plugin
	PVSelRowFilteringFunction_p ret = lib_rff->clone<PVSelRowFilteringFunction>();
	ret->set_combination_op((PVCore::PVBinaryOperation) op);

	// Get back the arguments
	PVCore::PVArgumentList args = PVCore::QDomElement_to_PVArgumentList(elt_args, ret->get_default_args());
	ret->set_args(args);

	return ret;
}
