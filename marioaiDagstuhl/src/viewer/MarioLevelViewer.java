package viewer;

import static reader.JsonReader.JsonToDoubleArray;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.Point;
import java.io.File;
import java.io.IOException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;

import javax.imageio.ImageIO;

import basicMap.Settings;
import ch.idsia.ai.tasks.ProgressTask;
import ch.idsia.mario.engine.LevelRenderer;
import ch.idsia.mario.engine.level.Level;
import ch.idsia.mario.engine.level.LevelParser;
import ch.idsia.tools.CmdLineOptions;
import ch.idsia.tools.EvaluationOptions;
import cmatest.MarioEvalFunction;
import reader.JsonReader;

/**
 * This file allows you to generate a level image for any latent vector
 * or your choice. The vector must have a length of 32 numbers separated
 * by commas enclosed in square brackets [ ]. For example,
 * [0.9881835842209917, -0.9986077315374948, 0.9995512051242508,
 * 0.9998643432807639, -0.9976165917284504, -0.9995247114230822,
 * -0.9997001909358728, 0.9995694511739592, -0.9431036754879115,
 * 0.9998155541290887, 0.9997863689962382, -0.8761392912669269,
 * -0.999843833016589, 0.9993230720045649, 0.9995470247917402,
 * -0.9998847606084427, -0.9998322053148382, 0.9997707200294411,
 * -0.9998905141832997, -0.9999512510490688, -0.9533512808031753,
 * 0.9997703088007039, -0.9992229823819915, 0.9953917828622341,
 * 0.9973473366437476, 0.9943030781608361, 0.9995290290713732,
 * -0.9994945079679955, 0.9997109900652238, -0.9988379572928884,
 * 0.9995070647543864, 0.9994132207570211]
 * 
 */
public class MarioLevelViewer {

	public static final int BLOCK_SIZE = 16;
	public static final int LEVEL_HEIGHT = 14;
	// Pipe tile constants (from LevelParser)
	public static final byte PIPE_TOP_LEFT = 10;
	public static final byte PIPE_TOP_RIGHT = 11;
	public static final byte PIPE_LEFT = 26;
	public static final byte PIPE_RIGHT = 27;
	public static final byte EMPTY_TILE = 0;
	// Add other tile constants if needed for more stats
	public static final byte GROUND = 9;
	public static final byte BREAKABLE = 16;
	public static final byte QUESTION = 21;

	/**
	 * Return an image of the level, excluding
	 * the background, Mario, and enemy sprites.
	 * 
	 * @param level
	 * @return
	 */
	public static BufferedImage getLevelImage(Level level, boolean excludeBufferRegion) {
		EvaluationOptions options = new CmdLineOptions(new String[0]);
		ProgressTask task = new ProgressTask(options);
		// Added to change level
		options.setLevel(level);
		task.setOptions(options);

		int relevantWidth = (level.width - (excludeBufferRegion ? 2 * LevelParser.BUFFER_WIDTH : 0)) * BLOCK_SIZE;
		BufferedImage image = new BufferedImage(relevantWidth, LEVEL_HEIGHT * BLOCK_SIZE, BufferedImage.TYPE_INT_RGB);
		// Skips buffer zones at start and end of level
		LevelRenderer.renderArea((Graphics2D) image.getGraphics(), level, 0, 0,
				excludeBufferRegion ? LevelParser.BUFFER_WIDTH * BLOCK_SIZE : 0, 0, relevantWidth,
				LEVEL_HEIGHT * BLOCK_SIZE);
		return image;
	}

	/**
	 * Save level as an image
	 * 
	 * @param level      Mario Level
	 * @param name       Filename, not including jpg extension
	 * @param clipBuffer Whether to exclude the buffer region we add to all levels
	 * @throws IOException
	 */
	public static void saveLevel(Level level, String name, boolean clipBuffer) throws IOException {
		BufferedImage image = getLevelImage(level, clipBuffer);

		File file = new File(name + ".jpg");
		ImageIO.write(image, "jpg", file);
		System.out.println("File saved: " + file);
	}

	/**
	 * Analyzes various features of the level, like pipes and floating enemies.
	 * 
	 * @param level The level to analyze.
	 * @return A Map containing statistics.
	 */
	public static Map<String, Number> analyzeLevelFeatures(Level level) {
		Map<String, Number> stats = new HashMap<>();
		int width = level.width;
		int height = level.height;

		int totalPipes = 0;
		int brokenPipes = 0;
		int pipesWithBottom = 0; // Count pipes that have at least one body segment
		int floatingPipes = 0; // Count pipes whose bottom is over empty space
		int totalEnemies = 0;
		int floatingEnemies = 0;
		int groundTiles = 0;
		int breakableTiles = 0;
		int questionBlocks = 0;
		int pipeTiles = 0;
		int floorGaps = 0;

		Set<Point> partOfAnyPipe = new HashSet<>();

		// --- Pipe Analysis ---
		for (int y = 0; y < height - 1; y++) {
			for (int x = 0; x < width - 1; x++) {
				byte tile = level.map[x][y];
				Point coord = new Point(x, y);
				int currentMaxY = -1; // Track lowest y of this specific pipe structure
				boolean hasBody = false;

				if (tile == PIPE_TOP_LEFT && !partOfAnyPipe.contains(coord)) {
					totalPipes++;
					currentMaxY = y; // Initial lowest point is the top
					Set<Point> currentPipeCoords = new HashSet<>();
					currentPipeCoords.add(coord);
					boolean isBroken = false;
					Point neighborCoord = new Point(x + 1, y);

					if (level.map[x + 1][y] != PIPE_TOP_RIGHT) {
						isBroken = true;
					} else {
						currentPipeCoords.add(neighborCoord);
						// Check body
						if (y + 1 >= height || level.map[x][y + 1] != PIPE_LEFT
								|| level.map[x + 1][y + 1] != PIPE_RIGHT) {
							isBroken = true;
							// Still add potential coords below if they exist
							if (y + 1 < height) {
								currentPipeCoords.add(new Point(x, y + 1));
								currentPipeCoords.add(new Point(x + 1, y + 1));
								// Even if broken here, check if these were pipe parts for maxY
								if (level.map[x][y + 1] == PIPE_LEFT || level.map[x + 1][y + 1] == PIPE_RIGHT) {
									currentMaxY = y + 1;
									hasBody = true;
								}
							}
						} else {
							// Valid first body segment
							hasBody = true;
							currentMaxY = y + 1;
							currentPipeCoords.add(new Point(x, y + 1));
							currentPipeCoords.add(new Point(x + 1, y + 1));
							// Check downwards
							int currentY = y + 2;
							while (currentY < height) {
								byte leftBody = level.map[x][currentY];
								byte rightBody = level.map[x + 1][currentY];
								boolean leftIsBody = (leftBody == PIPE_LEFT);
								boolean rightIsBody = (rightBody == PIPE_RIGHT);

								// Pipe ends if neither is a body part OR if one is body and other is non-empty,
								// non-body
								if ((!leftIsBody && !rightIsBody) ||
										(leftIsBody && rightBody != EMPTY_TILE && !rightIsBody) ||
										(rightIsBody && leftBody != EMPTY_TILE && !leftIsBody)) {
									break;
								}

								currentPipeCoords.add(new Point(x, currentY));
								currentPipeCoords.add(new Point(x + 1, currentY));
								currentMaxY = currentY; // Update lowest point

								// Check for inconsistencies (e.g., left without right on non-empty tile)
								if (!((leftIsBody && rightIsBody) ||
										(leftIsBody && rightBody == EMPTY_TILE) ||
										(leftIsBody && rightBody == PIPE_RIGHT) || // Allow right body if left is
																					// correct
										(rightIsBody && leftBody == EMPTY_TILE) ||
										(rightIsBody && leftBody == PIPE_LEFT))) { // Allow left body if right is
																					// correct
									isBroken = true;
									break;
								}
								currentY++;
							}
						}
					}
					if (isBroken)
						brokenPipes++;
					partOfAnyPipe.addAll(currentPipeCoords);

					// Check for floating *after* processing the whole pipe structure
					if (hasBody) {
						pipesWithBottom++;
						// Check below the lowest found body part (currentMaxY)
						if (currentMaxY + 1 < height) {
							if (level.map[x][currentMaxY + 1] == EMPTY_TILE
									&& level.map[x + 1][currentMaxY + 1] == EMPTY_TILE) {
								floatingPipes++;
							}
						} else {
							// Pipe reaches bottom of the level, considered floating
							floatingPipes++;
						}
					}

				}
				// Check for orphaned top-right (counts as a broken pipe)
				else if (tile == PIPE_TOP_RIGHT && !partOfAnyPipe.contains(coord)) {
					totalPipes++;
					brokenPipes++;
					partOfAnyPipe.add(coord);
					// Orphaned tops don't have a bottom and cannot float by this definition
				}
			}
		}
		// Check for orphaned body parts
		// int orphanedBrokenIncrement = 0; // Simpler: Don't try to avoid double
		// counting broken structures from orphans here
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				byte tile = level.map[x][y];
				Point coord = new Point(x, y);
				if ((tile == PIPE_LEFT || tile == PIPE_RIGHT) && !partOfAnyPipe.contains(coord)) {
					// An orphan body part implies *some* structure is broken.
					// Increment brokenPipes, but be aware this might overcount if multiple orphans
					// belong to the same conceptual broken pipe not caught by top-down check.
					// brokenPipes++; // Let's NOT increment brokenPipes here to avoid overcounting
					// based on previous runs. Rely on top-down check for broken count.
					partOfAnyPipe.add(coord); // Still mark as part of *some* pipe structure
				}
			}
		}

		// --- Enemy & Other Tile Analysis ---
		boolean inGap = false;
		for (int x = 0; x < width; x++) {
			// Floor gap check
			if (height > 0 && level.map[x][height - 1] == EMPTY_TILE) {
				if (!inGap) {
					floorGaps++;
					inGap = true;
				}
			} else {
				inGap = false;
			}

			for (int y = 0; y < height; y++) {
				// Enemy check
				if (level.spriteTemplates[x][y] != null) {
					totalEnemies++;
					// Check tile below for support
					if (y + 1 >= height || level.map[x][y + 1] == EMPTY_TILE) {
						floatingEnemies++;
					}
				}
				// Other tile counts
				byte tile = level.map[x][y];
				if (tile == GROUND)
					groundTiles++;
				else if (tile == BREAKABLE)
					breakableTiles++;
				else if (tile == QUESTION)
					questionBlocks++;
				else if (tile == PIPE_LEFT || tile == PIPE_RIGHT || tile == PIPE_TOP_LEFT || tile == PIPE_TOP_RIGHT)
					pipeTiles++;
			}
		}

		// Populate results map
		stats.put("LevelWidth", width);
		stats.put("LevelHeight", height);
		stats.put("TotalPipes", totalPipes);
		stats.put("BrokenPipes", brokenPipes);
		stats.put("PipesWithBottom", pipesWithBottom); // Pipes that could potentially float
		stats.put("FloatingPipes", floatingPipes); // Pipes confirmed floating
		stats.put("TotalEnemies", totalEnemies);
		stats.put("FloatingEnemies", floatingEnemies);
		stats.put("GroundTiles", groundTiles);
		stats.put("BreakableTiles", breakableTiles);
		stats.put("QuestionBlocks", questionBlocks);
		stats.put("PipeTiles", pipeTiles);
		stats.put("FloorGaps", floorGaps);

		// Derived stats
		int groundedPipes = pipesWithBottom - floatingPipes; // Pipes with bottom that are NOT floating
		double groundedPipePercentage = (pipesWithBottom > 0) ? ((double) groundedPipes / pipesWithBottom * 100.0)
				: 100.0; // Default to 100% if no pipes have a bottom section
		double levelValidPipePercentage = (totalPipes > 0) ? ((double) (totalPipes - brokenPipes) / totalPipes * 100.0)
				: 100.0;
		stats.put("LevelValidPipePercentage", levelValidPipePercentage);
		stats.put("GroundedPipes", groundedPipes); // Added
		stats.put("GroundedPipePercentage", groundedPipePercentage); // Added
		stats.put("LevelHasBrokenPipe", (brokenPipes > 0) ? 1 : 0);
		stats.put("LevelHasFloatingEnemy", (floatingEnemies > 0) ? 1 : 0);
		stats.put("LevelHasFloatingPipe", (floatingPipes > 0) ? 1 : 0); // Added

		return stats;
	}

	public static void main(String[] args) throws IOException {
		// Check system property to control file saving
		boolean saveFiles = Boolean.parseBoolean(System.getProperty("mariogan.savefiles", "true"));

		Settings.setPythonProgram();
		MarioEvalFunction eval = new MarioEvalFunction(false);

		Level level;
		// --- Level Generation Logic (Same as before) ---
		String strLatentVector = "";
		String[] commandLineArgs = args; // Store original args

		// Filter out the -D argument if present, before passing to level generation
		List<String> filteredArgsList = new ArrayList<>();
		for (String arg : args) {
			if (!arg.startsWith("-Dmariogan.savefiles")) { // Be specific about the property
				filteredArgsList.add(arg);
			}
		}
		String[] filteredArgs = filteredArgsList.toArray(new String[0]);

		if (filteredArgs.length > 0) {
			StringBuilder builder = new StringBuilder();
			for (String str : filteredArgs) { // Use filtered args here
				builder.append(str);
			}
			strLatentVector = builder.toString();
			// Settings.printInfoMsg("Passed vector(s): " + strLatentVector); // Suppress
			if (strLatentVector.subSequence(0, 2).equals("[[")) {
				// ... (multi-vector logic remains the same) ...
				strLatentVector = strLatentVector.substring(1, strLatentVector.length() - 1);
				String levels = "";
				while (strLatentVector.length() > 0) {
					int end = strLatentVector.indexOf("]") + 1;
					String oneVector = strLatentVector.substring(0, end);
					// System.out.println("ONE VECTOR: " + oneVector); // Suppress
					levels += eval.stringToFromGAN(oneVector); // Use the GAN
					strLatentVector = strLatentVector.substring(end); // discard processed vector
					if (strLatentVector.length() > 0) {
						levels += ",";
						strLatentVector = strLatentVector.substring(1); // discard leading comma
					}
				}
				levels = "[" + levels + "]"; // Put back in brackets
				// System.out.println(levels); // Suppress
				List<List<List<Integer>>> allLevels = JsonReader.JsonToInt(levels);
				ArrayList<List<Integer>> oneLevel = new ArrayList<List<Integer>>();
				for (List<Integer> row : allLevels.get(0)) {
					oneLevel.add(new ArrayList<Integer>());
				}
				for (List<List<Integer>> aLevel : allLevels) {
					int index = 0;
					for (List<Integer> row : aLevel) {
						oneLevel.get(index++).addAll(row);
					}
				}
				level = LevelParser.createLevelJson(oneLevel);
			} else {
				double[] latentVector = JsonToDoubleArray(strLatentVector);
				level = eval.levelFromLatentVector(latentVector);
			}
		} else {
			// System.out.println("No latent vector provided. Generating level with random
			// vector..."); // Suppress
			int latentDim = Integer.parseInt(Settings.GAN_DIM);
			double[] randomLatentVector = new double[latentDim];
			Random random = new Random();
			for (int i = 0; i < latentDim; i++) {
				randomLatentVector[i] = random.nextGaussian();
			}
			// System.out.println("Generated random vector: " +
			// java.util.Arrays.toString(randomLatentVector)); // Suppress
			level = eval.levelFromLatentVector(randomLatentVector);
		}
		// --- End Level Generation ---

		// Save files only if flag is set
		if (saveFiles) {
			try {
				saveLevel(level, "LevelClipped", true);
			} catch (Exception e) {
				System.err.println("Error saving LevelClipped.jpg: " + e.getMessage());
			}
			try {
				saveLevel(level, "LevelFull", false);
			} catch (Exception e) {
				System.err.println("Error saving LevelFull.jpg: " + e.getMessage());
			}
			try (PrintStream ps = new PrintStream(new FileOutputStream("level.txt"))) {
				level.saveText(ps);
			} catch (IOException e) {
				System.err.println("Error saving level.txt: " + e.getMessage());
			}
		}

		// --- Analyze and Print Stats ---
		Map<String, Number> stats = analyzeLevelFeatures(level);
		for (Map.Entry<String, Number> entry : stats.entrySet()) {
			if (entry.getValue() instanceof Double) {
				System.out.printf("%s: %.1f%n", entry.getKey(), entry.getValue());
			} else {
				System.out.printf("%s: %d%n", entry.getKey(), entry.getValue());
			}
		}

		eval.exit();
		System.exit(0);
	}
}
