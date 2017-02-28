/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author sathish
 */
import au.com.bytecode.opencsv.CSVReader;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import java.io.FileReader;
import java.io.IOException;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.List;
import java.util.Map;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

public final class CsvToVectors {

    private static long wordCount = 1;
    private static final Map<String, Long> words = Maps.newHashMap();
    private final String csvPath;

    public CsvToVectors(String csvPath) {
        this.csvPath = csvPath;
    }

    public List<MahoutVector> vectorize() throws IOException {

        List<MahoutVector> vectors = Lists.newArrayList();

        // Iterate the CSV records
        CSVReader reader = new CSVReader(new FileReader(this.csvPath));
        String[] line;

        try {
            while ((line = reader.readNext()) != null) {

                Vector vector = new RandomAccessSparseVector(line.length - 1, line.length - 1);
                int rowIndex = 0;

                // @attribute IR {P,A,N}
                vector.set(rowIndex, processString(line[rowIndex]));
                rowIndex++;

                // @attribute MR {P,A,N}
                vector.set(rowIndex, processString(line[rowIndex]));
                rowIndex++;

                // @attribute FF {P,A,N}
                vector.set(rowIndex, processString(line[rowIndex]));
                rowIndex++;

                //@attribute CR {P,A,N}
                vector.set(rowIndex, processString(line[rowIndex]));
                rowIndex++;

                //@attribute CO {P,A,N}
                vector.set(rowIndex, processString(line[rowIndex]));
                rowIndex++;

                //@attribute OP {P,A,N}
                vector.set(rowIndex, processString(line[rowIndex]));
                //rowIndex++;

                rowIndex = 6;
                // @attribute Class {B,NB} getting the predicting class
                String classifier = line[rowIndex];
                MahoutVector mahoutVector = new MahoutVector();
                mahoutVector.classifier = classifier;
                mahoutVector.vector = vector;
                mahoutVector.input = line[0]+","+line[1]+","+line[2]+","+line[3]+","+line[4]+","+ line[5];
                vectors.add(mahoutVector);
            }
            return vectors;
        } finally {
            reader.close();
        }
    }

    // Not sure how scalable this is going to be
    protected double processString(String data) {
        Long theLong = words.get(data);
        if (theLong == null) {
            theLong = wordCount++;
            words.put(data, theLong);
        }
        return theLong;
    }
/*
    protected double processNumeric(String data) {
        Double d = Double.NaN;
        if (isNumeric(data)) {
            d = Double.parseDouble(data);
        }

        return d;
    }

    public static boolean isNumeric(String str) {
        NumberFormat formatter = NumberFormat.getInstance();
        ParsePosition parsePosition = new ParsePosition(0);
        formatter.parse(str, parsePosition);
        return str.length() == parsePosition.getIndex();
    }
*/
}
