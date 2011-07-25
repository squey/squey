#include "PVFieldSplitterCSV.h"


// CSV classic delimiters
static char g_delimiters[] = {',',' ','\t',';','|'};

PVFilter::PVFieldSplitterCSV::PVFieldSplitterCSV(PVCore::PVArgumentList const& args)
{
	INIT_FILTER(PVFilter::PVFieldSplitterCSV, args);

}

void PVFilter::PVFieldSplitterCSV::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_sep = args["sep"].toChar().toAscii();
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterCSV)
{
	PVCore::PVArgumentList args;
	args["sep"] = QVariant(',');
	return args;
}

void PVFilter::PVFieldSplitterCSV::_csv_new_field(void* s, size_t len, void* p)
{
	// WARNING: this function will be called from different thread, so we can't use member
	// variables to store informations between different calls...
		
	// TOFIX: All of this is clearly suboptimal, too many memory copy occurs...
	// The ideal fix would be to have just an index and len to the original pointer !
	// (but this works)
	
	buf_infos *sp = (buf_infos*) p;
	QByteArray ba((const char*)s, len);
	QString str_field(ba); // convert to unicode

	size_t len_qs = str_field.size() * sizeof(QChar);
	size_t nchars = str_field.size();
	if (len_qs > sp->_len_buf) {
		PVLOG_WARN("(PVFieldSplitterCSV) chunk is not large enough to hold CSV datas !\n");
		return;
	}

	memcpy(sp->_f_cur, str_field.constData(), len_qs);
	PVCore::PVField f(*sp->_parent, (char*) (sp->_f_cur), (char*) (sp->_f_cur+nchars));

	sp->_f_cur += nchars;
	sp->_len_buf -= len_qs;

	PVLOG_HEAVYDEBUG("(PVFieldSplitterCSV): new CSV field: %s\n", qPrintable(f.qstr()));

	sp->_lf->insert(sp->_it_ins, f);
	sp->_nelts++;
}

void PVFilter::PVFieldSplitterCSV::_csv_new_row(int /*c*/, void* /*p*/)
{
	PVLOG_HEAVYDEBUG("(PVFieldSplitterCSV) in csv_new_row\n");
}

PVCore::list_fields::size_type PVFilter::PVFieldSplitterCSV::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	PVLOG_HEAVYDEBUG("(PVFieldSplitterCSV): in one_to_many\n");

	field.init_qstr();

	buf_infos inf;
	// Init csv parser
	if (csv_init(&inf._p, 0) != 0)
		PVLOG_ERROR("Unable to initialize libcsv !\n");
	csv_set_delim(&inf._p, _sep);

	// Convert to "C strings"
	inf._parent = field.elt_parent();
	inf._it_ins = it_ins;
	inf._lf = &l;
	inf._f_cur = (QChar*) field.begin();
	inf._len_buf = field.size();
	inf._cstr = field.qstr().toAscii();
	inf._nelts = 0;

	// And use libcsv
	csv_parse(&inf._p, inf._cstr.data(), inf._cstr.size(), &PVFilter::PVFieldSplitterCSV::_csv_new_field, &PVFilter::PVFieldSplitterCSV::_csv_new_row, &inf);
	csv_fini(&inf._p, &PVFilter::PVFieldSplitterCSV::_csv_new_field, &PVFilter::PVFieldSplitterCSV::_csv_new_row, &inf);
	csv_free(&inf._p);

	// Check that, if wanted to, the number of fields is the expected one.
	// If not, the element is invalid.
	
	PVLOG_DEBUG("(PVFieldsSplitterCSV) 0x%x: number expected fields: %d\n", this, _fields_expected);
	if (_fields_expected > 0 && _fields_expected != inf._nelts) {
		field.set_invalid();
		field.elt_parent()->set_invalid();
		return 0;
	}

	return inf._nelts;
}


bool PVFilter::PVFieldSplitterCSV::guess(list_guess_result_t& res, PVCore::PVField const& in_field)
{
	PVCore::PVArgumentList test_args;
	bool ok = false;

	for (size_t i = 0; i < sizeof(g_delimiters)/sizeof(char); i++) {
		PVCore::PVField own_field(in_field);
		own_field.deep_copy();
		PVCore::list_fields lf;
		test_args["sep"] = QVariant(QChar(g_delimiters[i]));
		set_args(test_args);
		if (one_to_many(lf, lf.begin(), own_field) > 1) {
			// We have a match
			res.push_back(list_guess_result_t::value_type(test_args, lf));
			ok = true;
		}
	}

	return ok;
}

IMPL_FILTER(PVFilter::PVFieldSplitterCSV)
