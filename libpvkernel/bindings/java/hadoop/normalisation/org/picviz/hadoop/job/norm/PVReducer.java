package org.picviz.hadoop.job.norm;

import java.io.IOException;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Reducer;

public class PVReducer extends Reducer<LongWritable, Text, LongWritable, Text> {
	public void reduce(LongWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
		for (Text v : values) {
			context.write(key, v);
		}
    }   
}
