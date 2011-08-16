#include <pvkernel/rush/PVChunkTransformUTF16.h>
#include <string>

#include <tbb/scalable_allocator.h>

PVRush::PVChunkTransformUTF16::PVChunkTransformUTF16()
{
	_mul_rs = 0.5; // By default, divide by 2

	// TODO: performance comparaison between the two libs !
#ifdef CHUNKTRANSFORM_USE_QT_UNICODE
	// A QTextDecoder is used because we can get partial characters as we're reading chunks.
	// It will handle this problem for us !
	_decoder = QTextCodec::codecForLocale()->makeDecoder();
#else // Use ICU
	UErrorCode status = U_ZERO_ERROR;
	_csd = ucsdet_open(&status);
	_ucnv = ucnv_open(NULL, &status);
	ucnv_resetToUnicode(_ucnv);
	_cd_found = false;
	_tmp_dest_size = 0;
	_tmp_dest = NULL;
#endif
}

PVRush::PVChunkTransformUTF16::~PVChunkTransformUTF16()
{
#ifdef CHUNKTRANSFORM_USE_QT_UNICODE
	delete _decoder;
#else
	ucsdet_close(_csd);
	ucnv_close(_ucnv);
	if (_tmp_dest) {
		static tbb::scalable_allocator<UChar> alloc;
		alloc.deallocate(_tmp_dest, _tmp_dest_size);
	}
#endif
}

size_t PVRush::PVChunkTransformUTF16::next_read_size(size_t org_size) const
{
	return ((size_t) ((float)org_size * _mul_rs));
}

size_t PVRush::PVChunkTransformUTF16::operator()(char* data, size_t len_read, size_t len_avail) const
{
#ifdef CHUNKTRANSFORM_USE_QT_UNICODE
	// For now, Qt is used. But using iconv could allow more control on memory,
	// and will potentially save one malloc from QString !
	
	// Find the good encoding. By default, we use Latin1.
	if (!_cd.found()) {
		if (_cd.HandleData(data, len_read) == NS_OK) {
			_cd.DataEnd();
			if (_cd.found()) {
				const std::string& cs = _cd.GetCharset();
				PVLOG_DEBUG("Encoding found : %s\n", cs.c_str());
				if (cs.find("UTF-16") != cs.npos)
					_mul_rs = 1;
				else
				if (cs.find("UTF-32") != cs.npos)
					_mul_rs = 1;
				QTextCodec* tc = QTextCodec::codecForName(cs.c_str());
				if (tc == NULL) {
					PVLOG_ERROR("Encoding %s not supported by Qt !\n", cs.c_str());
				}
				else
				{
					delete _decoder;
					_decoder = tc->makeDecoder();
				}
			}
		}
	}

	// Convert the chunk to UTF-16 thanks to Qt
	QString s_tmp = _decoder->toUnicode(data, len_read);
	const char* tmp_data = (const char*) s_tmp.unicode();
	const size_t final_size = (s_tmp.size())*2;

	if (final_size > len_avail) {
		PVLOG_ERROR("Size of chunk too small to get UTF16 datas ! (converted size: %d, available size: %ld)\n", final_size, len_avail);
		return len_read;
	}
	// And copy the result
	memcpy(data, tmp_data, final_size);
#else // Use ICU
	char* data_org = data;
	/*if (!_cd_found) {
		UErrorCode status = U_ZERO_ERROR;
		ucsdet_setText(_csd, data, len_read, &status);
		if (U_SUCCESS(status)) {
			const UCharsetMatch* ucm = ucsdet_detect(_csd, &status);
			if (U_SUCCESS(status)) {
				std::string cs(ucsdet_getName(ucm, &status));
				PVLOG_DEBUG("Encoding found : %s\n", cs.c_str());
				bool remove_bom = true;
				if (cs.find("UTF-16") != cs.npos)
					_mul_rs = 1;
				else
				if (cs.find("UTF-32") != cs.npos)
					_mul_rs = 1;
				else
				if (cs.find("UTF-8") == cs.npos)
					remove_bom = false;

				status = U_ZERO_ERROR;
				// Create ICU converter
				ucnv_close(_ucnv);
				_ucnv = ucnv_open(cs.c_str(), &status);
				if (U_FAILURE(status)) {
					PVLOG_ERROR("Encoding %s not supported by ICU ! Fall back to default decoder...\n", cs.c_str());
					status = U_ZERO_ERROR;
					_ucnv = ucnv_open(NULL, &status);
				}
				ucnv_resetToUnicode(_ucnv);
				_cd_found = true;

				if (remove_bom) {
					// Check first four bytes
					if (len_read >= 4) {
						uint32_t bom = *((uint32_t*)data);
						if (bom == 0xFFFE0000 || bom == 0x0000FEFF) {
							data += 4;
							len_read -= 4;
						}
					}
					if (len_read >= 3) {
						unsigned char* data_u = (unsigned char*) data;
						if (data_u[0] == 0xEF && data_u[1] == 0xBB && data_u[2] == 0xBF) {
							data += 3;
							len_read -= 3;
						}
					}
					if (len_read >= 2) {
						uint16_t& bom = *((uint16_t*)data);
						if (bom == 0xFFFE || bom == 0xFEFF) {
							data += 2;
							len_read -= 2;
						}
					}
				}
			}
		}
	}*/

	if (!_cd.found()) {
		if (_cd.HandleData(data, len_read) == NS_OK) {
			_cd.DataEnd();
			if (_cd.found()) {
				std::string cs(_cd.GetCharset());
				PVLOG_DEBUG("Encoding found : %s\n", cs.c_str());
				bool remove_bom = true;
				if (cs.find("UTF-16") != cs.npos)
					_mul_rs = 1;
				else
				if (cs.find("UTF-32") != cs.npos)
					_mul_rs = 1;
				else
				if (cs.find("UTF-8") == cs.npos)
					remove_bom = false;

				// Create ICU converter
				UErrorCode status = U_ZERO_ERROR;
				ucnv_close(_ucnv);
				_ucnv = ucnv_open(cs.c_str(), &status);
				if (U_FAILURE(status)) {
					PVLOG_ERROR("Encoding %s not supported by ICU ! Fall back to default decoder...\n", cs.c_str());
					status = U_ZERO_ERROR;
					_ucnv = ucnv_open(NULL, &status);
				}
				ucnv_resetToUnicode(_ucnv);
				_cd_found = true;

				if (remove_bom) {
					// Check first four bytes
					if (len_read >= 4) {
						uint32_t bom = *((uint32_t*)data);
						if (bom == 0xFFFE0000 || bom == 0x0000FEFF) {
							data += 4;
							len_read -= 4;
						}
					}
					if (len_read >= 3) {
						unsigned char* data_u = (unsigned char*) data;
						if (data_u[0] == 0xEF && data_u[1] == 0xBB && data_u[2] == 0xBF) {
							data += 3;
							len_read -= 3;
						}
					}
					if (len_read >= 2) {
						uint16_t& bom = *((uint16_t*)data);
						if (bom == 0xFFFE || bom == 0xFEFF) {
							data += 2;
							len_read -= 2;
						}
					}
				}
			}
		}
	}

	if (_tmp_dest_size < len_avail/sizeof(UChar)) {
		static tbb::scalable_allocator<UChar> alloc;
		if (_tmp_dest) {
			alloc.deallocate(_tmp_dest, _tmp_dest_size);
		}
		_tmp_dest_size = len_avail/sizeof(UChar);
		_tmp_dest = alloc.allocate(_tmp_dest_size);
	}

	// Convert the chunk to UTF-16 thanks to ICU
	UChar* target = _tmp_dest;
	const UChar* target_end = target + (len_avail/sizeof(UChar));
	const char* data_conv = data;
	const char* data_conv_end = data+len_read;
	UErrorCode status = U_ZERO_ERROR;
	ucnv_toUnicode(_ucnv, &target, target_end, &data_conv, data_conv_end, NULL, true, &status);
	const size_t final_size = (uintptr_t)target - (uintptr_t)_tmp_dest;
	if (status == U_BUFFER_OVERFLOW_ERROR) {
		PVLOG_ERROR("Size of chunk too small to get UTF16 datas ! (converted size: %d, available size: %ld)\n", final_size, len_avail);
		return len_read;
	}
	memcpy(data_org, _tmp_dest, final_size);

#endif

	return final_size;
}
