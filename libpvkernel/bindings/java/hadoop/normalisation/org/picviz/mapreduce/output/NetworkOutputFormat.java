package org.picviz.mapreduce.output;

import java.io.IOException;
import org.apache.hadoop.mapred.InvalidJobConfException;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.OutputCommitter;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

import java.net.InetAddress;
import java.net.UnknownHostException;

public abstract class NetworkOutputFormat<K, V> extends OutputFormat<K, V> {

	public static final String HOST = "mapreduce.output.networkoutputformat.host";

	public abstract RecordWriter<K, V> getRecordWriter(TaskAttemptContext job) throws IOException, InterruptedException;

	public void checkOutputSpecs(JobContext job) throws UnknownHostException, InvalidJobConfException {
		String host = getDestHost(job);
		if (host == null) {
			throw new InvalidJobConfException("Destination host not set.");
		}
		@SuppressWarnings("unused")
		InetAddress addr = InetAddress.getByName(host);
	}


	public static void setDestHost(JobContext job, String host) {
		job.getConfiguration().set(NetworkOutputFormat.HOST, host);
	}

	public static String getDestHost(JobContext job) {
		String host = job.getConfiguration().get(NetworkOutputFormat.HOST);
		return host;
	}
	
	@Override
	public abstract OutputCommitter getOutputCommitter(TaskAttemptContext arg0) throws IOException, InterruptedException ;
}

