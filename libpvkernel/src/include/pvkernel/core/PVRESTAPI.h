/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVKERNEL_CORE_PVRESTAPI_H__
#define __PVKERNEL_CORE_PVRESTAPI_H__

#ifdef INENDI_DEVELOPER_MODE
#define INSPECTOR_REST_API_DOMAIN "https://devapi.esi-inendi.com"
#define INSPECTOR_REST_API_PORT ":443"
#define INSPECTOR_REST_API_AUTH_TOKEN "1ae64b54eddfb65d65dc7da6d6891624f039e256"
#else // USER_TARGET=customer
#define INSPECTOR_REST_API_DOMAIN "https://api.esi-inendi.com"
#define INSPECTOR_REST_API_PORT ":443"
#define INSPECTOR_REST_API_AUTH_TOKEN "0ddb64e9821632d7f844ca059ff32b4f6a0618fc"
#endif
#define INSPECTOR_REST_API_SOCKET INSPECTOR_REST_API_DOMAIN INSPECTOR_REST_API_PORT

#endif // __PVKERNEL_CORE_PVRESTAPI_H__
