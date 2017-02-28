/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import java.io.IOException;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;

import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.naivebayes.ComplementaryNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author sathish
 */
public final class MahoutTest {
    private static final Logger logger = LoggerFactory.getLogger(MahoutTest.class);
    
    private static final String sequenceFile = "seq/part-0000";
    private static final String modelDirectory = "/tmp/classifier/model";
    private static final String labelIndexDirectory = "/tmp/classifier/label";
    
    private static final Configuration conf = new Configuration();
    
    public static void sequencer(String inputDataPath) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        Path seqFilePath = new Path(sequenceFile);

        fs.delete(seqFilePath, true);

        SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, seqFilePath, Text.class, VectorWritable.class);

        String csvPath = inputDataPath;

        try {
            CsvToVectors csvToVectors = new CsvToVectors(csvPath);
            List<MahoutVector> vectors = csvToVectors.vectorize();
            // Init the labels
            for (MahoutVector vector : vectors) {
                VectorWritable vectorWritable = new VectorWritable();
                vectorWritable.set(vector.vector);
                writer.append(new Text("/" + vector.classifier + "/"), vectorWritable);
            }
        }
        finally {
            writer.close();
        }
    }

    public static void train() throws Throwable {
        FileSystem fs = FileSystem.getLocal(conf);
        
        TrainNaiveBayesJob trainNaiveBayes = new TrainNaiveBayesJob();
        trainNaiveBayes.setConf(conf);
        
        fs.delete(new Path(modelDirectory), true);
        fs.delete(new Path(labelIndexDirectory), true);
        
        trainNaiveBayes.run(new String[]{"--input", sequenceFile, "--output", modelDirectory, "--tempDir", labelIndexDirectory, "--overwrite", "-el"});
        
    }
    
    public static void predict(String trainDataPath) throws Throwable {
        FileSystem fs = FileSystem.getLocal(conf);
   
        // Train the classifier
        NaiveBayesModel naiveBayesModel = NaiveBayesModel.materialize(new Path(modelDirectory), conf);
        System.out.println();
        System.out.print("    ");
        System.out.print("Features: " + (int) naiveBayesModel.numFeatures());
        System.out.print("\t\t");
        System.out.println("Labels: " + (int) naiveBayesModel.numLabels());
        System.out.println();
        
        AbstractVectorClassifier classifier = new ComplementaryNaiveBayesClassifier(naiveBayesModel);
        
        String csvPath = trainDataPath;

        CsvToVectors csvToVectors = new CsvToVectors(csvPath);
        List<MahoutVector> vectors = csvToVectors.vectorize();
                
        int total = 0;
        int success = 0;
        
        System.out.println("\tAnomaly\t\t\t" + "Normal\t\t\t" + "Classified\t" + "Class");
        System.out.println("-------------------------------------------------------------------------------------");
        for (MahoutVector mahoutVector : vectors){
            Vector prediction = classifier.classifyFull(mahoutVector.vector);
            
            // They sorted alphabetically
            double anomaly = prediction.get(0);
            double normal = prediction.get(1);

            String predictedClass = "NB";
            if (normal > anomaly) {
                predictedClass = "B";
            }
            
            System.out.print("\t");
            System.out.print(String.format("%.15f", anomaly));
            System.out.print("\t");
            System.out.print(String.format("%.15f", normal));
            System.out.print("\t    ");
            System.out.println(predictedClass + "\t\t  " + mahoutVector.classifier);
            
	    if (predictedClass.equals(mahoutVector.classifier))
	    {
                success++;
	    }
	    total ++;
        }
        System.out.println("-------------------------------------------------------------------------------------");
        System.out.println();
        System.out.print("    ");
        System.out.print("Total : " + total + "\t\t");
        System.out.print("Success : " + success + "\t\t");
        System.out.print("Failures : " + (total- success) + "\t\t");
        System.out.println("Ratio : " + (double) success/total);
    }
    
    public static void main(String[] args) throws Throwable {
        
        Options options = new Options();

        Option input = new Option("i", "input", true, "input data set");
        input.setRequired(true);
        options.addOption(input);

        Option output = new Option("t", "train", true, "train data set");
        output.setRequired(true);
        options.addOption(output);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("utility-name", options);
            System.exit(1);
            return;
        }

        String inputDataPath = cmd.getOptionValue("input");
        String trainDataPath = cmd.getOptionValue("train");
        
        sequencer(inputDataPath);
        train();
        predict(trainDataPath);
    }
}