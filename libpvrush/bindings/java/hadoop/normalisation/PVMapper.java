package org.picviz.hadoop.job.norm;

import java.io.IOException;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import org.apache.commons.lang.StringUtils;

import PVRushJNI;

public class PVMapper extends Mapper<LongWritable, Text, LongWritable, Text> {
	private PVRushJNI jni;

	public void map(LongWritable key, Text v, Context context) throws IOException, InterruptedException {
		String[] arr = jni.process_elt(v);
		if (arr.length == 0) {
			return;
		}

		context.write(key, StringUtils.join(arr, "\t"));
	}

	public void setup(Context context) {
		String s = context.getConfiguration().get("mapreduce.pvjob.format_path");
		jni.init_with_format(s);
	}   
}

