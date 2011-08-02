package org.picviz.hadoop.job.norm;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class PVJob {
	public static void main(String[] args) throws Exception {
		Configuration cfg = new Configuration();
		cfg.set("mapreduce.pvjob.format_path", args[2]);

		Job job = new Job(cfg);
		job.setJarByClass(PVJob.class); 
		
		job.setJobName("pvjob_normalisation");

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		job.setMapperClass(PVMapper.class);
		job.setNumReduceTasks(0);

		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(Text.class);

		job.waitForCompletion(true);
	}
}

