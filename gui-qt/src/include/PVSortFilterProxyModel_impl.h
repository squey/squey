#ifndef PVINSPECTOR_PVSORTFILTERPROXYMODEL_IMPL_H
#define PVINSPECTOR_PVSORTFILTERPROXYMODEL_IMPL_H

namespace PVInspector {

namespace __impl {

struct PVSortProxyAsc
{
	PVSortProxyAsc(PVSortFilterProxyModel* m_, int col): column(col), m(m_)
	{
		ms = m->sourceModel();
	}
	bool operator()(int idx1, int idx2) const
	{
		return m->less_than(ms->index(idx1, column), ms->index(idx2, column));
	}
private:
	int column;
	PVSortFilterProxyModel* m;
	QAbstractItemModel* ms;
};

struct PVSortProxyDesc
{
	PVSortProxyDesc(PVSortFilterProxyModel* m_, int col): column(col), m(m_)
	{
		ms = m->sourceModel();
	}
	bool operator()(int idx1, int idx2) const
	{
		return m->less_than(ms->index(idx2, column), ms->index(idx1, column));
	}
private:
	int column;
	PVSortFilterProxyModel* m;
	QAbstractItemModel* ms;
};

struct PVSortProxyComp
{
	PVSortProxyComp(PVSortFilterProxyModel* m_, int col): column(col), m(m_)
	{
		ms = m->sourceModel();
	}
	bool operator()(int idx1, int idx2) const
	{
		return m->is_equal(ms->index(idx2, column), ms->index(idx1, column));
	}
private:
	int column;
	PVSortFilterProxyModel* m;
	QAbstractItemModel* ms;
};

}

}

#endif
