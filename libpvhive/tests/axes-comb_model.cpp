#include "axes-comb_model.h"

void ModelIndexObserver::refresh()
{
	//emit const_cast<AxesCombinationListModel*>(_parent)->dataChanged(_parent->index(_row, 0), _parent->index(_row, 0));
	PVLOG_INFO("refresh %d\n", _parent->persistentIndexList().size());
}
