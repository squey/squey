package org.picviz.hadoop.job.norm;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Cluster;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.conf.Configured;

import org.picviz.mapreduce.output.NRAWNetworkOutputFormat;
import org.picviz.mapreduce.output.TCPNetworkOutputFormat;

public class PVJob extends Configured implements Tool {

	public int run(String[] args) throws Exception {
		Job job = Job.getInstance(new Cluster(getConf()), getConf());
		job.getConfiguration().set("mapreduce.pvjob.format_path", args[2]);

		job.setJarByClass(PVJob.class); 
		
		job.setJobName("pvjob_normalisation");

		FileInputFormat.addInputPath(job, new Path(args[0]));
		//FileOutputFormat.setOutputPath(job, new Path(args[1]));
		TCPNetworkOutputFormat.setDestPort(job, 1245);
		TCPNetworkOutputFormat.setDestHost(job, "172.16.0.250");

		job.setMapperClass(PVMapper.class);
		job.setNumReduceTasks(0);
		
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(ArrayWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(ArrayWritable.class);
		job.setOutputFormatClass(NRAWNetworkOutputFormat.class);
		
		job.waitForCompletion(true);
		
		// OutputCommitter.commitJob is never called, so that's a hack !
		TCPNetworkOutputFormat.sendLastFinishedTask("172.16.0.250", 1245);
		
		return 0;
	}

	public static int main(String[] args) throws Exception {
		int ret = ToolRunner.run(new Configuration(), new PVJob(), args);
		return ret;
	}
}
