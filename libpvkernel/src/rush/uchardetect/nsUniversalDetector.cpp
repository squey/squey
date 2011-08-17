/* -*- Mode: C++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* ***** BEGIN LICENSE BLOCK *****
 * Version: MPL 1.1/GPL 2.0/LGPL 2.1
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 * The Original Code is Mozilla Universal charset detector code.
 *
 * The Initial Developer of the Original Code is
 * Netscape Communications Corporation.
 * Portions created by the Initial Developer are Copyright (C) 2001
 * the Initial Developer. All Rights Reserved.
 *
 * Contributor(s):
 *          Shy Shalom <shooshX@gmail.com>
 *
 * Alternatively, the contents of this file may be used under the terms of
 * either the GNU General Public License Version 2 or later (the "GPL"), or
 * the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
 * in which case the provisions of the GPL or the LGPL are applicable instead
 * of those above. If you wish to allow use of your version of this file only
 * under the terms of either the GPL or the LGPL, and not to allow others to
 * use your version of this file under the terms of the MPL, indicate your
 * decision by deleting the provisions above and replace them with the notice
 * and other provisions required by the GPL or the LGPL. If you do not delete
 * the provisions above, a recipient may use your version of this file under
 * the terms of any one of the MPL, the GPL or the LGPL.
 *
 * ***** END LICENSE BLOCK ***** */

#include <pvkernel/rush/uchardetect/nscore.h>

#include <pvkernel/rush/uchardetect/universalchardet.h>
#include <pvkernel/rush/uchardetect/nsUniversalDetector.h>

#include <pvkernel/rush/uchardetect/nsMBCSGroupProber.h>
#include <pvkernel/rush/uchardetect/nsSBCSGroupProber.h>
#include <pvkernel/rush/uchardetect/nsEscCharsetProber.h>
#include <pvkernel/rush/uchardetect/nsLatin1Prober.h>

#include <pvkernel/core/picviz_intrin.h>

nsUniversalDetector::nsUniversalDetector()
{
  mDone = PR_FALSE;
  mBestGuess = -1;   //illegal value as signal
  mInTag = PR_FALSE;
  mEscCharSetProber = nsnull;

  mStart = PR_TRUE;
  mDetectedCharset = nsnull;
  mGotData = PR_FALSE;
  mInputState = ePureAscii;
  mLastChar = '\0';

  PRUint32 i;
  for (i = 0; i < NUM_OF_CHARSET_PROBERS; i++)
    mCharSetProbers[i] = nsnull;
}

nsUniversalDetector::~nsUniversalDetector() 
{
  for (PRInt32 i = 0; i < NUM_OF_CHARSET_PROBERS; i++)
    if (mCharSetProbers[i])      
      delete mCharSetProbers[i];
  if (mEscCharSetProber)
    delete mEscCharSetProber;
}

void 
nsUniversalDetector::Reset()
{
  mDone = PR_FALSE;
  mBestGuess = -1;   //illegal value as signal
  mInTag = PR_FALSE;

  mStart = PR_TRUE;
  mDetectedCharset = nsnull;
  mGotData = PR_FALSE;
  mInputState = ePureAscii;
  mLastChar = '\0';

  if (mEscCharSetProber)
    mEscCharSetProber->Reset();

  PRUint32 i;
  for (i = 0; i < NUM_OF_CHARSET_PROBERS; i++)
    if (mCharSetProbers[i])
      mCharSetProbers[i]->Reset();
}

//---------------------------------------------------------------------
#define SHORTCUT_THRESHOLD      (float)0.95
#define MINIMUM_THRESHOLD      (float)0.20

#ifdef __SSE4_2__
nsresult nsUniversalDetector::HandleData(const char* aBuf, PRUint32 aLen)
{
	if(mDone) 
		return NS_OK;

	if (aLen > 0)
		mGotData = PR_TRUE;

	//If the data starts with BOM, we know it is UTF
	if (mStart)
	{
		mStart = PR_FALSE;
		if (aLen > 3)
			switch (aBuf[0])
			{
				case '\xEF':
					if (('\xBB' == aBuf[1]) && ('\xBF' == aBuf[2]))
						// EF BB BF  UTF-8 encoded BOM
						mDetectedCharset = CHARDET_ENCODING_UTF_8;
					break;
				case '\xFE':
					if (('\xFF' == aBuf[1]) && ('\x00' == aBuf[2]) && ('\x00' == aBuf[3]))
						// FE FF 00 00  UCS-4, unusual octet order BOM (3412)
						mDetectedCharset = CHARDET_ENCODING_X_ISO_10646_UCS_4_3412;
					else if ('\xFF' == aBuf[1])
						// FE FF  UTF-16, big endian BOM
						mDetectedCharset = CHARDET_ENCODING_UTF_16BE;
					break;
				case '\x00':
					if (('\x00' == aBuf[1]) && ('\xFE' == aBuf[2]) && ('\xFF' == aBuf[3]))
						// 00 00 FE FF  UTF-32, big-endian BOM
						mDetectedCharset = CHARDET_ENCODING_UTF_32BE;
					else if (('\x00' == aBuf[1]) && ('\xFF' == aBuf[2]) && ('\xFE' == aBuf[3]))
						// 00 00 FF FE  UCS-4, unusual octet order BOM (2143)
						mDetectedCharset = CHARDET_ENCODING_X_ISO_10646_UCS_4_2143;
					break;
				case '\xFF':
					if (('\xFE' == aBuf[1]) && ('\x00' == aBuf[2]) && ('\x00' == aBuf[3]))
						// FF FE 00 00  UTF-32, little-endian BOM
						mDetectedCharset = CHARDET_ENCODING_UTF_32LE;
					else if ('\xFE' == aBuf[1])
						// FF FE  UTF-16, little endian BOM
						mDetectedCharset = CHARDET_ENCODING_UTF_16LE;
					break;
			}  // switch

		if (mDetectedCharset)
		{
			mDone = PR_TRUE;
			return NS_OK;
		}
	}

	PRUint32 i;
	PRUint32 nsse = aLen >> 4; // nsse = aLen/(16/sizeof(char));
	__m128i sse_str;
	const __m128i sse_high = _mm_set1_epi8('\x80');
	const __m128i sse_a0 = _mm_set1_epi8('\xA0');
	//const __m128i sse_033 = _mm_set1_epi8('\033');
	//const __m128i sse_ctrl = _mm_set1_epi16(0x7b7e); // == '~{'
	const __m128i sse_ff = _mm_set1_epi32(0xFFFFFFFF);
	const __m128i sse_rightshift = _mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
	char* aBufSSE = (char*) aBuf;
	for (i = 0; i < nsse; i++)
	{
		sse_str = _mm_loadu_si128((__m128i*) aBufSSE);
		//other than 0xa0, if every othe character is ascii, the page is ascii
		int test_ascii = _mm_testz_si128(sse_str, sse_high);
		// If testz_si128 returns 1, it means that sse_str & sse_high == 0, so that we have plain ascii.
		if (test_ascii == 0) {
			// Else, check that all bytes above 0x80 are equals to 0xA0
			__m128i mask_nonascii,mask_a0;
			mask_nonascii = _mm_cmpgt_epi8(sse_str, sse_high);
			mask_a0 = _mm_cmpeq_epi8(sse_str, sse_a0);
			// _mm_testc will return 1 if and only if all bits set in mask_nonascii are set in mask_a0 (which means that, if a[i] >= 0x80, a[i] == 0x80)
			if (_mm_testc_si128(mask_a0, mask_nonascii) == 0)
			{
				//we got a non-ascii byte (high-byte)
				//adjust state
				mInputState = eHighbyte;
				break;
			}
		}
		// Disable these character sets
		/*if (ePureAscii == mInputState)
		{
			//ok, just pure ascii so far
			// Search for \033 or '~{'
			__m128i sse_cmp = _mm_cmpeq_epi8(sse_str, sse_033);
			if (!_mm_testz_si128(sse_cmp, sse_ff)) {
				mInputState = eEscAscii;
			}
			else {
				sse_cmp = _mm_cmpeq_epi16(sse_str, sse_ctrl);
				int pos_ctrl = _mm_cmpestri(sse_ctrl, 2, sse_str, 16, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT);
				if ((pos_ctrl < 16) || (mLastChar == '~' && *aBufSSE == '{'))
					mInputState = eEscAscii;
			}
		}*/
		
		//mLastChar = aBufSSE[15];
		aBufSSE += 16;
	}
	if (ePureAscii == mInputState) // Prologue
	{
		char* aBufEnd = (char*) aBuf + aLen;
		while (aBufSSE != aBufEnd)
		{
			char c = *aBufSSE;
			if (c & '\x80' && c != '\xA0')  //Since many Ascii only page contains NBSP 
			{
				mInputState = eHighbyte;
				break;
			}
			/*if (c == '\033' || (c == '{' && mLastChar == '~')) {
				mInputState = eEscAscii;
			}*/
			//mLastChar = c;
			aBufSSE++;
		}
	}
	if (mInputState == eHighbyte) {
		//kill mEscCharSetProber if it is active
		/*if (mEscCharSetProber) {
			delete mEscCharSetProber;
			mEscCharSetProber = nsnull;
		}*/

		//start multibyte and singlebyte charset prober
		if (nsnull == mCharSetProbers[0])
			mCharSetProbers[0] = new nsMBCSGroupProber;
		if (nsnull == mCharSetProbers[1])
			mCharSetProbers[1] = new nsSBCSGroupProber;
		if (nsnull == mCharSetProbers[2])
			mCharSetProbers[2] = new nsLatin1Prober; 

		if ((nsnull == mCharSetProbers[0]) ||
				(nsnull == mCharSetProbers[1]) ||
				(nsnull == mCharSetProbers[2]))
			return NS_ERROR_OUT_OF_MEMORY;
	}

	nsProbingState st;
	switch (mInputState)
	{
		/*case eEscAscii:
			if (nsnull == mEscCharSetProber) {
				mEscCharSetProber = new nsEscCharSetProber;
				if (nsnull == mEscCharSetProber)
					return NS_ERROR_OUT_OF_MEMORY;
			}
			st = mEscCharSetProber->HandleData(aBuf, aLen);
			if (st == eFoundIt)
			{
				mDone = PR_TRUE;
				mDetectedCharset = mEscCharSetProber->GetCharSetName();
			}
			break;*/
		case eHighbyte:
			for (i = 0; i < NUM_OF_CHARSET_PROBERS; i++)
			{
				st = mCharSetProbers[i]->HandleData(aBuf, aLen);
				if (st == eFoundIt) 
				{
					mDone = PR_TRUE;
					mDetectedCharset = mCharSetProbers[i]->GetCharSetName();
					return NS_OK;
				} 
			}
			break;

		default:  //pure ascii
			;//do nothing here
	}
	return NS_OK;
}

#else
#warning nsUniversalDetector: SSE4.2 not enabled, nsUniversalDetector::HandleData will use the sequential version.
nsresult nsUniversalDetector::HandleData(const char* aBuf, PRUint32 aLen)
{
  if(mDone) 
    return NS_OK;

  if (aLen > 0)
    mGotData = PR_TRUE;

  //If the data starts with BOM, we know it is UTF
  if (mStart)
  {
    mStart = PR_FALSE;
    if (aLen > 3)
      switch (aBuf[0])
        {
        case '\xEF':
          if (('\xBB' == aBuf[1]) && ('\xBF' == aBuf[2]))
            // EF BB BF  UTF-8 encoded BOM
            mDetectedCharset = CHARDET_ENCODING_UTF_8;
        break;
        case '\xFE':
          if (('\xFF' == aBuf[1]) && ('\x00' == aBuf[2]) && ('\x00' == aBuf[3]))
            // FE FF 00 00  UCS-4, unusual octet order BOM (3412)
            mDetectedCharset = CHARDET_ENCODING_X_ISO_10646_UCS_4_3412;
          else if ('\xFF' == aBuf[1])
            // FE FF  UTF-16, big endian BOM
            mDetectedCharset = CHARDET_ENCODING_UTF_16BE;
        break;
        case '\x00':
          if (('\x00' == aBuf[1]) && ('\xFE' == aBuf[2]) && ('\xFF' == aBuf[3]))
            // 00 00 FE FF  UTF-32, big-endian BOM
            mDetectedCharset = CHARDET_ENCODING_UTF_32BE;
          else if (('\x00' == aBuf[1]) && ('\xFF' == aBuf[2]) && ('\xFE' == aBuf[3]))
            // 00 00 FF FE  UCS-4, unusual octet order BOM (2143)
            mDetectedCharset = CHARDET_ENCODING_X_ISO_10646_UCS_4_2143;
        break;
        case '\xFF':
          if (('\xFE' == aBuf[1]) && ('\x00' == aBuf[2]) && ('\x00' == aBuf[3]))
            // FF FE 00 00  UTF-32, little-endian BOM
            mDetectedCharset = CHARDET_ENCODING_UTF_32LE;
          else if ('\xFE' == aBuf[1])
            // FF FE  UTF-16, little endian BOM
            mDetectedCharset = CHARDET_ENCODING_UTF_16LE;
        break;
      }  // switch

      if (mDetectedCharset)
      {
        mDone = PR_TRUE;
        return NS_OK;
      }
  }
  
  PRUint32 i;
  for (i = 0; i < aLen; i++)
  {
    //other than 0xa0, if every othe character is ascii, the page is ascii
    if (aBuf[i] & '\x80' && aBuf[i] != '\xA0')  //Since many Ascii only page contains NBSP 
    {
      //we got a non-ascii byte (high-byte)
      if (mInputState != eHighbyte)
      {
        //adjust state
        mInputState = eHighbyte;

        //kill mEscCharSetProber if it is active
        if (mEscCharSetProber) {
          delete mEscCharSetProber;
          mEscCharSetProber = nsnull;
        }

        //start multibyte and singlebyte charset prober
        if (nsnull == mCharSetProbers[0])
          mCharSetProbers[0] = new nsMBCSGroupProber;
        if (nsnull == mCharSetProbers[1])
          mCharSetProbers[1] = new nsSBCSGroupProber;
        if (nsnull == mCharSetProbers[2])
          mCharSetProbers[2] = new nsLatin1Prober; 

        if ((nsnull == mCharSetProbers[0]) ||
            (nsnull == mCharSetProbers[1]) ||
            (nsnull == mCharSetProbers[2]))
            return NS_ERROR_OUT_OF_MEMORY;
      }
    }
    else
    {
      //ok, just pure ascii so far
      if ( ePureAscii == mInputState &&
        (aBuf[i] == '\033' || (aBuf[i] == '{' && mLastChar == '~')) )
      {
        //found escape character or HZ "~{"
        mInputState = eEscAscii;
      }
      mLastChar = aBuf[i];
    }
  }

  nsProbingState st;
  switch (mInputState)
  {
  case eEscAscii:
    if (nsnull == mEscCharSetProber) {
      mEscCharSetProber = new nsEscCharSetProber;
      if (nsnull == mEscCharSetProber)
        return NS_ERROR_OUT_OF_MEMORY;
    }
    st = mEscCharSetProber->HandleData(aBuf, aLen);
    if (st == eFoundIt)
    {
      mDone = PR_TRUE;
      mDetectedCharset = mEscCharSetProber->GetCharSetName();
    }
    break;
  case eHighbyte:
    for (i = 0; i < NUM_OF_CHARSET_PROBERS; i++)
    {
      st = mCharSetProbers[i]->HandleData(aBuf, aLen);
      if (st == eFoundIt) 
      {
        mDone = PR_TRUE;
        mDetectedCharset = mCharSetProbers[i]->GetCharSetName();
        return NS_OK;
      } 
    }
    break;

  default:  //pure ascii
    ;//do nothing here
  }
  return NS_OK;
}
#endif


//---------------------------------------------------------------------
void nsUniversalDetector::DataEnd()
{
  if (!mGotData)
  {
    // we haven't got any data yet, return immediately 
    // caller program sometimes call DataEnd before anything has been sent to detector
    return;
  }

  if (mDetectedCharset)
  {
    mDone = PR_TRUE;
    Report(mDetectedCharset);
    return;
  }
  
  switch (mInputState)
  {
  case eHighbyte:
    {
      float proberConfidence;
      float maxProberConfidence = (float)0.0;
      PRInt32 maxProber = 0;

      for (PRInt32 i = 0; i < NUM_OF_CHARSET_PROBERS; i++)
      {
        proberConfidence = mCharSetProbers[i]->GetConfidence();
        if (proberConfidence > maxProberConfidence)
        {
          maxProberConfidence = proberConfidence;
          maxProber = i;
        }
      }
      //do not report anything because we are not confident of it, that's in fact a negative answer
      if (maxProberConfidence > MINIMUM_THRESHOLD)
        Report(mCharSetProbers[maxProber]->GetCharSetName());
    }
    break;
  case eEscAscii:
    break;
  default:
    ;
  }
  return;
}
