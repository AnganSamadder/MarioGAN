package reader;

import ch.idsia.mario.engine.GlobalOptions;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonSyntaxException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Provides utility methods for reading level data and latent vectors from JSON.
 */
public class JsonReader {
	private List<List<List<Integer>>> levelsJson; // Changed name for clarity
	int currentLevelIndex;

	/**
	 * Constructor that reads level data from a file or string based on
	 * GlobalOptions.
	 * 
	 * @param source Path to the JSON file or the JSON string itself.
	 */
	public JsonReader(String source) {
		if (GlobalOptions.JsonAsString) {
			levelsJson = jsonStringToIntLevels(source);
		} else {
			levelsJson = jsonFileToIntLevels(source);
		}
		currentLevelIndex = 0;
	}

	/**
	 * @return The total number of levels loaded.
	 */
	public int getNumberOfLevels() {
		return (levelsJson != null) ? levelsJson.size() : 0;
	}

	/**
	 * Retrieves a specific level by its index.
	 * 
	 * @param index The index of the level.
	 * @return The level data as a List<List<Integer>>, or null if index is invalid.
	 */
	public List<List<Integer>> getLevel(int index) {
		if (levelsJson != null && index >= 0 && index < levelsJson.size()) {
			return levelsJson.get(index);
		} else {
			System.err.println("Error: Invalid level index requested: " + index);
			return null;
		}
	}

	/**
	 * @return True if there are more levels available to iterate through.
	 */
	public boolean hasNextLevel() {
		return (levelsJson != null && currentLevelIndex < levelsJson.size());
	}

	/**
	 * Retrieves the next level in the sequence and advances the iterator.
	 * 
	 * @return The next level data, or null if no more levels are available.
	 */
	public List<List<Integer>> nextLevel() {
		if (hasNextLevel()) {
			return levelsJson.get(currentLevelIndex++);
		} else {
			return null;
		}
	}

	/**
	 * Reads a JSON file containing multiple levels and converts it to a list of
	 * integer levels.
	 * Each level is represented as a List<List<Integer>>.
	 * 
	 * @param fileLocation Path to the JSON file.
	 * @return List of levels, or an empty list if reading/parsing fails.
	 */
	public static List<List<List<Integer>>> jsonFileToIntLevels(String fileLocation) {
		try {
			List<String> lines = Files.readAllLines(Paths.get(fileLocation), Charset.defaultCharset());
			StringBuilder jsonStringBuilder = new StringBuilder();
			for (String s : lines) { // Use StringBuilder for efficiency
				jsonStringBuilder.append(s);
			}
			return jsonStringToIntLevels(jsonStringBuilder.toString());
		} catch (IOException e) {
			System.err.println("Error reading JSON file '" + fileLocation + "': " + e.getMessage());
			return new ArrayList<>(); // Return empty list on error
		}
	}

	/**
	 * Converts a JSON string representing multiple levels into a list of integer
	 * levels.
	 * 
	 * @param jsonString The JSON string.
	 * @return List of levels, or an empty list if parsing fails.
	 */
	public static List<List<List<Integer>>> jsonStringToIntLevels(String jsonString) {
		List<List<List<Integer>>> levelList = new ArrayList<>();
		if (jsonString == null || jsonString.trim().isEmpty()) {
			System.err.println("Error: Input JSON string is null or empty.");
			return levelList;
		}

		try {
			JsonArray topLevelArray = new Gson().fromJson(jsonString, JsonArray.class);
			if (topLevelArray == null) {
				System.err.println("Error: Failed to parse top-level JSON array.");
				return levelList;
			}

			for (JsonElement levelElement : topLevelArray) {
				if (!levelElement.isJsonArray())
					continue; // Skip non-array elements
				JsonArray levelArray = levelElement.getAsJsonArray();
				List<List<Integer>> currentLevel = new ArrayList<>();
				for (JsonElement rowElement : levelArray) {
					if (!rowElement.isJsonArray())
						continue; // Skip non-array rows
					JsonArray rowArray = rowElement.getAsJsonArray();
					List<Integer> currentRow = new ArrayList<>();
					for (JsonElement tileElement : rowArray) {
						if (tileElement.isJsonPrimitive() && tileElement.getAsJsonPrimitive().isNumber()) {
							currentRow.add(tileElement.getAsNumber().intValue());
						} else {
							System.err.println("Warning: Non-numeric JSON element encountered in level data: "
									+ tileElement.toString() + ". Replacing with 0.");
							currentRow.add(0); // Default value for non-numeric elements
						}
					}
					currentLevel.add(currentRow);
				}
				levelList.add(currentLevel);
			}
		} catch (JsonSyntaxException e) {
			System.err.println("Error parsing JSON string: " + e.getMessage());
			// Return potentially partially parsed list or empty list
		} catch (Exception e) { // Catch other potential runtime errors
			System.err.println("Unexpected error processing JSON string: " + e.getMessage());
		}
		return levelList;
	}

	/**
	 * Converts a JSON string representing a single array of numbers into a double
	 * array.
	 * 
	 * @param jsonString The JSON string, e.g., "[1.0, -0.5, 0.0]".
	 * @return A double array, or an empty array if parsing fails.
	 */
	public static double[] JsonToDoubleArray(String jsonString) {
		if (jsonString == null || jsonString.trim().isEmpty()) {
			System.err.println("Error: Input JSON string for double array is null or empty.");
			return new double[0];
		}
		try {
			JsonArray jsonArray = new Gson().fromJson(jsonString, JsonArray.class);
			if (jsonArray == null) {
				System.err.println("Error: Failed to parse JSON array for double array.");
				return new double[0];
			}
			double[] doubleArray = new double[jsonArray.size()];
			for (int i = 0; i < jsonArray.size(); i++) {
				JsonElement element = jsonArray.get(i);
				if (element.isJsonPrimitive() && element.getAsJsonPrimitive().isNumber()) {
					doubleArray[i] = element.getAsDouble();
				} else {
					System.err.println("Warning: Non-numeric JSON element encountered in double array: "
							+ element.toString() + ". Replacing with 0.0.");
					doubleArray[i] = 0.0;
				}
			}
			return doubleArray;
		} catch (JsonSyntaxException e) {
			System.err.println("Error parsing JSON string for double array: " + e.getMessage());
			return new double[0];
		} catch (Exception e) {
			System.err.println("Unexpected error processing JSON string for double array: " + e.getMessage());
			return new double[0];
		}
	}

	// Method previously causing errors - keep commented or remove if truly unused
	/*
	 * public static List<List<List<Integer>>> generateLevels(List<double[]>
	 * latentVectors, Integer latentDim) {
	 * System.err.
	 * println("Error: generateLevels method is likely incorrect or deprecated.");
	 * // This method likely needs to interact with a Python script via
	 * MarioEvalFunction
	 * // or similar mechanism. It cannot directly generate levels in Java.
	 * return new ArrayList<>();
	 * }
	 */
}
