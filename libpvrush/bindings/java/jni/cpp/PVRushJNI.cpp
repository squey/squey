#include <jni.h>
#include "PVRushJNI.h"

#include <pvcore/PVElement.h>
#include <pvfilter/PVElementFilter.h>
#include <pvfilter/PVPluginsLoad.h>
#include <pvrush/PVFormat.h>
#include <pvrush/PVPluginsLoad.h>

#include <QString>

#include <iostream>

PVRush::PVFormat _format;
PVFilter::PVElementFilter_f _elt_f;

// Helper functions
static QString jstring_to_qstring(JNIEnv* env, jstring str)
{
	const jchar* buf_utf16 = env->GetStringChars(str, NULL);
	int n = env->GetStringLength(str);
	std::cout << "Size of string: " << n << std::endl;
	return QString::fromRawData((const QChar*) buf_utf16, n);
}

static void free_j2qstring(JNIEnv* env, jstring str, QString const& qstr)
{
	env->ReleaseStringChars(str, (const jchar*) qstr.constData());
}

// JNI interface
JNIEXPORT void JNICALL Java_PVRushJNI_init_1with_1format(JNIEnv * env, jobject /*obj*/, jstring str)
{
	std::cout << std::endl;
	std::cerr << "hello\nhi\n";
	PVRush::PVPluginsLoad::load_all_plugins();
	PVFilter::PVPluginsLoad::load_all_plugins();

	QString file_path = jstring_to_qstring(env, str);
	std::cout << "Path to format: " << qPrintable(file_path) << std::endl;
	_format.populate_from_xml(file_path);
	std::cout << "Populate done" << std::endl;
	_elt_f = _format.create_tbb_filters_elt();
	free_j2qstring(env, str, file_path);
}

JNIEXPORT jobjectArray JNICALL Java_PVRushJNI_process_1elt(JNIEnv *env, jobject /*obj*/, jstring elt)
{
	jclass str_class = env->FindClass("java/lang/String");
	QString qelt = jstring_to_qstring(env, elt);
	PVCore::PVElement pvelt(NULL, (char*) qelt.constData(), (char*) qelt.constData()+qelt.size()*sizeof(QChar));
	_elt_f(pvelt);

	PVCore::list_fields const& lf = pvelt.c_fields();

	// Check that the element is still valid
	size_t nfields = lf.size();
	if (nfields == 0 || !pvelt.valid()) {
		return NULL;
	}

	// Create the returned array
	jstring empty_str = env->NewString((const jchar*) "\0\0", 0);
	jobjectArray list_ret = env->NewObjectArray(nfields, str_class, empty_str);
	PVCore::list_fields::const_iterator it;
	size_t i = 0;
	for (it = lf.begin(); it != lf.end(); it++) {
		PVCore::PVField const& field = *it;
		jstring jf = env->NewString((const jchar*) field.begin(), field.size() / sizeof(QChar));
		env->SetObjectArrayElement(list_ret, i, jf);
		i++;
	}

	free_j2qstring(env, elt, qelt);
	return list_ret;
}
