#include <pvkernel/rush/PVChunkAlignBoundary.h>


PVRush::PVChunkAlignBoundary::PVChunkAlignBoundary(QTextBoundaryFinder::BoundaryType boundary = QTextBoundaryFinder::Line) :
	_b(boundary)
{
}


bool PVRush::PVChunkAlignBoundary::operator()(PVCore::PVChunk &cur_chunk, PVCore::PVChunk &next_chunk)
{
	QString str_tmp = QString::fromRawData((QChar*) cur_chunk.begin(), cur_chunk.size()/sizeof(QChar));
	QTextBoundaryFinder finder(_b, str_tmp);
	int previous = 0;
	int next = finder.toNextBoundary();
	if (next == 0) {
		previous = 1;
		next = finder.toNextBoundary();
	}

	unsigned int nelts = 0;
	QChar* str_start = (QChar*) cur_chunk.begin();
	while (next != -1) {
		QChar* start = str_start + previous;
		QChar* end = str_start + next;

		cur_chunk.add_element((char*) start, (char*) end);
		nelts++;

		previous = next + 1;
		next = finder.toNextBoundary();
	}
	if (nelts == 0) {
		PVLOG_WARN("PVChunkAlignBoundary: boundary hasn't been found !\n");
		return false;
	}
	cur_chunk.set_end((char*) (str_start+previous-1));

	// What's remaining goes to the next_chunk
	size_t snc = sizeof(QChar)*(str_tmp.size() - previous);
	memcpy(next_chunk.begin(), (char*) (str_start + previous), snc);
	next_chunk.set_end(next_chunk.begin() + snc);
	return true;
}
