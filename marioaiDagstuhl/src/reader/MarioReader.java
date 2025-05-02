package reader;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
// import com.google.gson.stream.JsonWriter; // Unused

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException; // Added
import java.io.PrintWriter;
import java.util.*;

// import static basicMap.Settings.printErrorMsg; // Assuming these are not needed or defined elsewhere
// import static basicMap.Settings.printWarnMsg;

/**
 * Reads Mario level files (.txt format) and converts them into integer arrays,
 * optionally creating JSON representations suitable for GAN training.
 */
public class MarioReader {

    // Mapping from character representation in .txt files to integer tile IDs.
    static final Map<Character, Integer> TILE_MAP = new HashMap<>();

    static {
        TILE_MAP.put('X', 0); // Solid block (Brick? Ground? Check usage)
        TILE_MAP.put('S', 1); // Breakable block (?)
        TILE_MAP.put('-', 2); // Empty space (background)
        TILE_MAP.put('?', 3); // Question block (coin)
        TILE_MAP.put('Q', 4); // Question block (powerup)
        TILE_MAP.put('E', 5); // Enemy placeholder (actual type defined elsewhere)
        TILE_MAP.put('<', 6); // Pipe top-left
        TILE_MAP.put('>', 7); // Pipe top-right
        TILE_MAP.put('[', 8); // Pipe body left
        TILE_MAP.put(']', 9); // Pipe body right
        TILE_MAP.put('o', 10); // Coin
        // B and b: Bullet Bill cannon parts
        TILE_MAP.put('B', 11); // Cannon top
        TILE_MAP.put('b', 12); // Cannon body/base
    }

    // Target width for level segments used in training.
    static int TARGET_SEGMENT_WIDTH = 28;
    static int UNKNOWN_TILE_ID = 2; // Default to empty space if tile char is unknown

    /**
     * Main method for processing a directory of level files.
     * Reads all .txt files, converts them to integer arrays, generates training
     * segments,
     * and saves the results as JSON files (one per original level, and one combined
     * file).
     * 
     * @param args Command line arguments (currently unused, paths are hardcoded).
     * @throws Exception If file I/O errors occur.
     */
    public static void main(String[] args) throws Exception {
        // --- Configuration (Consider making these command-line arguments) ---
        String workingDir = System.getProperty("user.dir");
        // Assuming marioaiDagstuhl is a subfolder in the working directory
        String projectBaseDir = workingDir; // Adjust if marioaiDagstuhl is elsewhere
        String inputDirectoryPath = projectBaseDir + "/marioaiDagstuhl/data/mario/levels/"; // Relative path assumed
        String outputDirectoryPath = inputDirectoryPath; // Output segments to the same dir
        String combinedJsonOutputPath = outputDirectoryPath + "levels_combined.json"; // Combined output file
        // ---------------------------------------------------------------------

        System.out.println("Working Directory: " + workingDir);
        System.out.println("Input Level Directory: " + inputDirectoryPath);
        System.out.println("Output Directory: " + outputDirectoryPath);

        File inputDirectory = new File(inputDirectoryPath);
        if (!inputDirectory.isDirectory()) {
            System.err.println("Error: Input path is not a directory: " + inputDirectoryPath);
            return;
        }

        File outputDirectory = new File(outputDirectoryPath);
        if (!outputDirectory.exists()) {
            System.out.println("Creating output directory: " + outputDirectoryPath);
            if (!outputDirectory.mkdirs()) {
                System.err.println("Error: Could not create output directory.");
                return;
            }
        }

        // List to hold all segments from all valid levels
        List<int[][]> allSegments = new ArrayList<>();
        Gson gson = new GsonBuilder().setPrettyPrinting().create(); // Use pretty printing for readability

        File[] levelFiles = inputDirectory.listFiles((dir, name) -> name.toLowerCase().endsWith(".txt"));

        if (levelFiles == null || levelFiles.length == 0) {
            System.err.println("Warning: No .txt level files found in " + inputDirectoryPath);
            return;
        }

        System.out.println("Found " + levelFiles.length + " level files to process.");

        for (File levelFile : levelFiles) {
            String inputFilename = levelFile.getName();
            System.out.println("\n--- Processing: " + inputFilename + " ---");
            try (FileInputStream fis = new FileInputStream(levelFile); Scanner scanner = new Scanner(fis)) {

                int[][] level = readLevel(scanner);
                if (level == null || level.length == 0 || level[0].length <= TARGET_SEGMENT_WIDTH) {
                    System.out.println("  Skipping level (too short or read error).");
                    continue;
                }

                List<int[][]> segments = createSegments(level, TARGET_SEGMENT_WIDTH);
                if (segments.isEmpty()) {
                    System.out.println("  No segments generated (level width might be exactly target width?).");
                    continue;
                }

                allSegments.addAll(segments);
                System.out.println("  Extracted " + segments.size() + " segments.");

                // Save segments for this individual level as JSON
                String baseName = inputFilename.substring(0, inputFilename.lastIndexOf('.'));
                String individualJsonPath = outputDirectoryPath + "segments_" + baseName + ".json";
                try (FileWriter fileWriter = new FileWriter(individualJsonPath);
                        PrintWriter printWriter = new PrintWriter(fileWriter)) {
                    gson.toJson(segments, printWriter);
                    System.out.println("  Saved individual segments to: " + individualJsonPath);
                } catch (IOException e) {
                    System.err.println("  Error saving individual JSON for " + inputFilename + ": " + e.getMessage());
                }

            } catch (Exception e) {
                System.err.println("  Error processing file " + inputFilename + ": " + e.getMessage());
                e.printStackTrace(); // Print stack trace for debugging
            }
        }

        // Save all combined segments to a single JSON file
        if (!allSegments.isEmpty()) {
            System.out.println("\n--- Saving Combined Segments --- ");
            try (FileWriter fileWriter = new FileWriter(combinedJsonOutputPath);
                    PrintWriter printWriter = new PrintWriter(fileWriter)) {
                gson.toJson(allSegments, printWriter);
                System.out.println(
                        "Saved combined segments (" + allSegments.size() + " total) to: " + combinedJsonOutputPath);
            } catch (IOException e) {
                System.err.println("Error saving combined JSON file: " + e.getMessage());
            }
        } else {
            System.out.println("\nNo segments were generated from any level file.");
        }

        System.out.println("\n--- Processing Finished ---");
    }

    /**
     * Creates overlapping segments of a specified width from a full level.
     * 
     * @param level        The full level represented as a 2D integer array.
     * @param segmentWidth The desired width of each segment.
     * @return A List of 2D integer arrays, each representing a level segment.
     */
    static List<int[][]> createSegments(int[][] level, int segmentWidth) {
        List<int[][]> segments = new ArrayList<>();
        if (level == null || level.length == 0 || level[0].length <= segmentWidth) {
            // Return empty list if level is invalid or not wider than segment width
            return segments;
        }

        int h = level.length;
        int w = level[0].length;

        // Slide a window of segmentWidth across the level
        for (int offset = 0; offset <= w - segmentWidth; offset++) {
            int[][] segment = new int[h][segmentWidth];
            for (int y = 0; y < h; y++) {
                // Copy the relevant portion of the row
                System.arraycopy(level[y], offset, segment[y], 0, segmentWidth);
            }
            segments.add(segment);
        }
        return segments;
    }

    /**
     * Reads a level from a text file representation using a Scanner.
     * 
     * @param scanner The Scanner object initialized with the level file stream.
     * @return A 2D integer array representing the level, or null on error.
     * @throws Exception If errors occur during reading.
     */
    static int[][] readLevel(Scanner scanner) throws Exception {
        List<String> lines = new ArrayList<>();
        int width = -1; // Initialize width to -1 to detect inconsistent line lengths

        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            if (line.trim().isEmpty())
                continue; // Skip empty lines

            if (width == -1) { // First non-empty line determines width
                width = line.length();
            } else if (line.length() != width) {
                System.err.println("Error: Inconsistent line length found. Expected " + width + ", got " + line.length()
                        + " for line: " + line);
                // Optionally, throw an exception or handle differently
                return null; // Indicate error by returning null
            }
            lines.add(line);
        }

        if (lines.isEmpty() || width <= 0) {
            System.err.println("Error: Level file is empty or contains no valid level data.");
            return null;
        }

        int height = lines.size();
        int[][] levelArray = new int[height][width];
        System.out.println("  Level dimensions: " + width + "x" + height);

        for (int y = 0; y < height; y++) {
            String line = lines.get(y);
            for (int x = 0; x < width; x++) {
                char tileChar = line.charAt(x);
                Integer tileId = TILE_MAP.get(tileChar);
                if (tileId != null) {
                    levelArray[y][x] = tileId;
                } else {
                    // Handle unknown characters - print warning and use default ID
                    System.err.println(
                            String.format("  Warning: Unknown character '%c' at (%d, %d). Replacing with tile ID %d.",
                                    tileChar, x, y, UNKNOWN_TILE_ID));
                    levelArray[y][x] = UNKNOWN_TILE_ID;
                }
            }
        }
        return levelArray;
    }

    // --- Helper for debugging (removed from main logic) ---
    /*
     * static String arrayToString(int[][] inputArray) {
     * // ... (Implementation can be kept for debugging if needed) ...
     * }
     */
}
