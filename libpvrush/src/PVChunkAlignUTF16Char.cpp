#include <pvrush/PVChunkAlignUTF16Char.h>


PVRush::PVChunkAlignUTF16Char::PVChunkAlignUTF16Char(QChar c) :
	_c(c)
{
}


bool PVRush::PVChunkAlignUTF16Char::operator()(PVCore::PVChunk &cur_chunk, PVCore::PVChunk &next_chunk)
{
	QString str_tmp = QString::fromRawData((QChar*) cur_chunk.begin(), cur_chunk.size()/sizeof(QChar));
	// Special case when _c is a t the beggining of the chunk
	int old_pos_c = 0;
	if (str_tmp[0] == _c)
		old_pos_c = 1;
	QChar* str_start = (QChar*) cur_chunk.begin();
	int pos_c;
	unsigned int nelts = 0;
	PVCore::list_elts &elts = cur_chunk.elements();
	// QString::size() here does not include a terminating null character, because the string have been
	// created with QString::fromRawData, and the originating datas didn't have a tareminating null character
	ssize_t sstr = str_tmp.size();
	while ((pos_c = str_tmp.indexOf(_c, old_pos_c)) != -1) {
		QChar* start = str_start + old_pos_c;
		QChar* end = str_start + pos_c;

		elts.push_back(PVCore::PVElement(&cur_chunk, (char*)start, (char*)end));
		nelts++;
	
		old_pos_c = pos_c+1;
	}
	if (nelts == 0) {
		PVLOG_WARN("PVChunkAlignUTF16Char: character hasn't been found !\n");
		return false;
	}
	cur_chunk.set_end((char*) (str_start+old_pos_c-1));

	// What's remaining goes to the next_chunk	
	memcpy(next_chunk.begin(), (char*) (str_start + old_pos_c), sizeof(QChar)*(sstr - old_pos_c));
	next_chunk.set_end(next_chunk.begin() + sizeof(QChar)*(sstr-old_pos_c));


	return true;
}
