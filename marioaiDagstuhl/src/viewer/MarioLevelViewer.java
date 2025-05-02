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
import java.io.InputStream;
import java.io.DataInputStream;
import java.util.Queue;
import java.util.LinkedList;
import java.io.FileInputStream;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;

import javax.imageio.ImageIO;

// Imports for A* simulation
import ch.idsia.ai.agents.Agent;
import ch.idsia.tools.EvaluationInfo;
import ch.idsia.tools.Evaluator;
import ch.idsia.mario.engine.sprites.Mario; // Needed for STATUS_WIN
import competition.icegic.robin.AStarAgent; // Assuming this is the correct path

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
 * Generates level images and optionally runs simulations or analysis.
 * Can be controlled via command-line arguments.
 */
public class MarioLevelViewer {

	public static final int BLOCK_SIZE = 16;
	public static final int LEVEL_HEIGHT = 14;

	// Tile constants (derived from LevelParser or common usage)
	public static final byte PIPE_TOP_LEFT = 10;
	public static final byte PIPE_TOP_RIGHT = 11;
	public static final byte PIPE_LEFT = 26;
	public static final byte PIPE_RIGHT = 27;
	public static final byte EMPTY_TILE = 0;
	public static final byte GROUND = 9;
	public static final byte BREAKABLE = 16;
	public static final byte QUESTION = 21;
	public static final byte GOOMBA = (byte) 80; // Example enemy ID
	// Add other enemy IDs if needed: WINGED_GOOMBA = (byte) 95, etc.
	public static final Set<Byte> ENEMY_TILES = new HashSet<>(Arrays.asList(GOOMBA));
	// Define blocking tiles for covered pipe check
	public static final Set<Byte> BLOCKING_TILES = new HashSet<>(Arrays.asList(
			GROUND, BREAKABLE, QUESTION // Add any other solid, non-passable blocks here
	));

	/**
	 * Renders an image of the level structure (tiles).
	 * 
	 * @param level               The Level object.
	 * @param excludeBufferRegion Whether to clip the buffer zones at the start/end.
	 * @return A BufferedImage of the level.
	 */
	public static BufferedImage getLevelImage(Level level, boolean excludeBufferRegion) {
		EvaluationOptions options = new CmdLineOptions(new String[0]);
		ProgressTask task = new ProgressTask(options);
		options.setLevel(level);
		task.setOptions(options);

		int bufferWidthPixels = LevelParser.BUFFER_WIDTH * BLOCK_SIZE;
		int startX = excludeBufferRegion ? bufferWidthPixels : 0;
		int renderWidth = level.width * BLOCK_SIZE - (excludeBufferRegion ? 2 * bufferWidthPixels : 0);
		int renderHeight = level.height * BLOCK_SIZE; // Use actual level height

		// Ensure render width is positive
		if (renderWidth <= 0) {
			System.err.println(
					"Warning: Render width is zero or negative. Level might be too small or buffer clipping too large.");
			renderWidth = BLOCK_SIZE; // Render at least one block
		}

		BufferedImage image = new BufferedImage(renderWidth, renderHeight, BufferedImage.TYPE_INT_RGB);
		Graphics2D g = (Graphics2D) image.getGraphics();

		// Render the specified area
		LevelRenderer.renderArea(g, level, 0, 0, startX, 0, renderWidth, renderHeight);
		g.dispose();
		return image;
	}

	/**
	 * Saves the rendered level image to a file.
	 * 
	 * @param level      The Level object.
	 * @param filename   Base filename (e.g., "level_01"). Extension ".jpg" added
	 *                   automatically.
	 * @param clipBuffer Whether to exclude the buffer region.
	 * @throws IOException If file saving fails.
	 */
	public static void saveLevelImage(Level level, String filename, boolean clipBuffer) throws IOException {
		BufferedImage image = getLevelImage(level, clipBuffer);
		File file = new File(filename + ".jpg");
		ImageIO.write(image, "jpg", file);
		System.out.println("Level image saved: " + file.getAbsolutePath());
	}

	/**
	 * Analyzes various structural features of the level.
	 * 
	 * @param level The level to analyze.
	 * @return A Map containing calculated statistics.
	 */
	public static Map<String, Number> analyzeLevelFeatures(Level level) {
		Map<String, Number> stats = new HashMap<>();
		int width = level.width;
		int height = level.height;

		int totalPipeStructures = 0;
		int brokenPipeStructures = 0;
		int floatingPipeStructures = 0;
		int coveredPipeStructures = 0;
		int totalEnemies = 0;
		int floatingEnemies = 0;
		int floorGaps = 0;
		int totalPipeTiles = 0; // Count individual pipe tiles

		Set<Point> visitedPipeTiles = new HashSet<>(); // Keep track of visited tiles during BFS

		// --- Analyze Tiles and Enemies ---
		for (int y = 0; y < height; y++) {
			boolean floorStarted = false;
			boolean inGap = false;
			for (int x = 0; x < width; x++) {
				byte tile = level.map[x][y];
				Point currentPoint = new Point(x, y);

				// Count enemy types
				if (ENEMY_TILES.contains(level.spriteTemplates[x][y])) { // Check sprite template layer
					totalEnemies++;
					// Check for floating enemies (simplified: no solid block directly below)
					if (y < height - 1) {
						byte tileBelow = level.map[x][y + 1];
						// Add more robust check if needed (e.g., allow standing on pipes?)
						if (tileBelow == EMPTY_TILE) {
							floatingEnemies++;
						}
					} else { // Enemy in the bottom row is considered floating
						floatingEnemies++;
					}
				}

				// Check for floor gaps (only on the bottom-most row)
				if (y == height - 1) {
					boolean isGroundLike = (tile == GROUND || tile == BREAKABLE); // Define what constitutes floor
					if (isGroundLike) {
						floorStarted = true;
						inGap = false;
					} else if (floorStarted && !inGap) {
						floorGaps++;
						inGap = true;
					}
				}

				// --- Pipe Structure Analysis using BFS ---
				boolean isPipeTile = (tile == PIPE_TOP_LEFT || tile == PIPE_TOP_RIGHT ||
						tile == PIPE_LEFT || tile == PIPE_RIGHT);
				if (isPipeTile)
					totalPipeTiles++; // Count individual tiles

				if (isPipeTile && !visitedPipeTiles.contains(currentPoint)) {
					totalPipeStructures++;
					Queue<Point> queue = new LinkedList<>();
					Set<Point> currentPipeCoords = new HashSet<>();
					boolean structureHasBody = false;
					int structureMaxY = -1;
					boolean structureIsCovered = false;
					boolean structureIsFloating = true; // Assume floating initially
					boolean structureIsBroken = false;

					queue.add(currentPoint);
					visitedPipeTiles.add(currentPoint);
					currentPipeCoords.add(currentPoint);

					// BFS traversal
					while (!queue.isEmpty()) {
						Point p = queue.poll();
						int px = p.x;
						int py = p.y;
						byte pTile = level.map[px][py];

						structureMaxY = Math.max(structureMaxY, py);
						if (pTile == PIPE_LEFT || pTile == PIPE_RIGHT)
							structureHasBody = true;

						// Check if pipe top is covered
						if ((pTile == PIPE_TOP_LEFT || pTile == PIPE_TOP_RIGHT) && py > 0 &&
								BLOCKING_TILES.contains(level.map[px][py - 1])) {
							structureIsCovered = true;
						}

						// Check for grounding (only for tiles at max Y)
						if (py == structureMaxY && py < height - 1 &&
								(level.map[px][py + 1] == GROUND || level.map[px][py + 1] == BREAKABLE)) {
							structureIsFloating = false; // Found grounding
						}

						// Check neighbors for BFS
						int[] dx = { 0, 0, 1, -1 };
						int[] dy = { 1, -1, 0, 0 };
						for (int i = 0; i < 4; i++) {
							int nx = px + dx[i];
							int ny = py + dy[i];
							Point neighborPoint = new Point(nx, ny);

							if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
									!visitedPipeTiles.contains(neighborPoint)) {
								byte neighborTile = level.map[nx][ny];
								boolean neighborIsPipe = (neighborTile == PIPE_TOP_LEFT ||
										neighborTile == PIPE_TOP_RIGHT ||
										neighborTile == PIPE_LEFT ||
										neighborTile == PIPE_RIGHT);
								if (neighborIsPipe) {
									visitedPipeTiles.add(neighborPoint);
									currentPipeCoords.add(neighborPoint);
									queue.add(neighborPoint);
								}
							}
						}
					} // End BFS

					// --- Structure Analysis ---
					boolean hasLeft = currentPipeCoords.stream()
							.anyMatch(p -> level.map[p.x][p.y] == PIPE_TOP_LEFT || level.map[p.x][p.y] == PIPE_LEFT);
					boolean hasRight = currentPipeCoords.stream()
							.anyMatch(p -> level.map[p.x][p.y] == PIPE_TOP_RIGHT || level.map[p.x][p.y] == PIPE_RIGHT);
					boolean hasTop = currentPipeCoords.stream().anyMatch(
							p -> level.map[p.x][p.y] == PIPE_TOP_LEFT || level.map[p.x][p.y] == PIPE_TOP_RIGHT);

					// Broken if: missing left/right pair OR has body but no top
					structureIsBroken = !(hasLeft && hasRight) || (structureHasBody && !hasTop);

					// If the structure reached the bottom row during BFS, it can't be floating.
					if (structureMaxY == height - 1)
						structureIsFloating = false;

					if (structureIsBroken)
						brokenPipeStructures++;
					if (structureIsFloating)
						floatingPipeStructures++;
					if (structureIsCovered)
						coveredPipeStructures++;
				}
			}
		}

		stats.put("totalPipeStructures", totalPipeStructures);
		stats.put("brokenPipeStructures", brokenPipeStructures);
		stats.put("floatingPipeStructures", floatingPipeStructures);
		stats.put("coveredPipeStructures", coveredPipeStructures);
		stats.put("totalEnemies", totalEnemies);
		stats.put("floatingEnemies", floatingEnemies);
		stats.put("floorGaps", floorGaps);
		stats.put("totalPipeTiles", totalPipeTiles); // Added stat
		// Add other basic counts if needed
		// stats.put("groundTiles", ...);
		// stats.put("breakableTiles", ...);
		// stats.put("questionBlocks", ...);

		return stats;
	}

	/**
	 * Runs an A* agent simulation on the level.
	 * 
	 * @param level     The Level object.
	 * @param visualize Whether to show the simulation GUI.
	 * @return EvaluationInfo containing simulation results (null if simulation
	 *         fails).
	 */
	public static EvaluationInfo simulateLevel(Level level, boolean visualize) {
		EvaluationOptions options = new CmdLineOptions(new String[0]);
		options.setLevel(level);
		options.setVisualization(visualize);
		// options.setFPS(24); // Adjust FPS if needed
		Agent agent = new AStarAgent(); // Use the specific A* agent
		options.setAgent(agent);
		Evaluator evaluator = new Evaluator(options);
		try {
			List<EvaluationInfo> results = evaluator.evaluate();
			if (results != null && !results.isEmpty()) {
				return results.get(0); // Return info from the first (and likely only) run
			} else {
				System.err.println("Error: Simulation failed or returned no results.");
				return null;
			}
		} catch (Exception e) {
			System.err.println("Exception during A* simulation:");
			e.printStackTrace();
			return null;
		}
	}

	// --- Main Method ---
	public static void main(String[] args) throws IOException {
		// Default parameters
		String latentVectorString = null;
		String outputDir = "output_levels";
		String checkpointPath = null; // No default, use MARIOGAN_CHECKPOINT or fail
		boolean simulate = false;
		boolean analyze = false;
		boolean saveImage = true;
		boolean visualizeSim = false;
		int numLevels = 1;
		String filenamePrefix = "level";
		boolean clipBuffer = true;
		Integer latentDim = null; // Auto-detect if possible

		// Argument Parsing
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
				case "--vector":
				case "-v":
					if (i + 1 < args.length) {
						latentVectorString = args[++i];
					} else {
						System.err.println("Error: --vector requires a string argument.");
						return;
					}
					break;
				case "--outputDir":
				case "-o":
					if (i + 1 < args.length) {
						outputDir = args[++i];
					} else {
						System.err.println("Error: --outputDir requires a path argument.");
						return;
					}
					break;
				case "--checkpoint":
				case "-c":
					if (i + 1 < args.length) {
						checkpointPath = args[++i];
					} else {
						System.err.println("Error: --checkpoint requires a path argument.");
						return;
					}
					break;
				case "--numLevels":
				case "-n":
					if (i + 1 < args.length) {
						try {
							numLevels = Integer.parseInt(args[++i]);
						} catch (NumberFormatException e) {
							System.err.println("Error: Invalid number for --numLevels.");
							return;
						}
					} else {
						System.err.println("Error: --numLevels requires an integer argument.");
						return;
					}
					break;
				case "--latentDim":
				case "-nz":
					if (i + 1 < args.length) {
						try {
							latentDim = Integer.parseInt(args[++i]);
						} catch (NumberFormatException e) {
							System.err.println("Error: Invalid number for --latentDim.");
							return;
						}
					} else {
						System.err.println("Error: --latentDim requires an integer argument.");
						return;
					}
					break;
				case "--prefix":
					if (i + 1 < args.length) {
						filenamePrefix = args[++i];
					} else {
						System.err.println("Error: --prefix requires a string argument.");
						return;
					}
					break;
				case "--simulate":
					simulate = true;
					break;
				case "--analyze":
					analyze = true;
					break;
				case "--visualizeSim":
					visualizeSim = true;
					break;
				case "--noImage":
					saveImage = false;
					break;
				case "--keepBuffer":
					clipBuffer = false;
					break;
				default:
					System.err.println("Unknown argument: " + args[i]);
			}
		}

		// Determine final checkpoint path
		String finalCheckpointPath = checkpointPath; // Use CLI arg if provided
		if (finalCheckpointPath == null) {
			finalCheckpointPath = System.getenv("MARIOGAN_CHECKPOINT"); // Fallback to env var
		}
		if (finalCheckpointPath == null) {
			System.err.println(
					"Error: Checkpoint path must be provided via --checkpoint or MARIOGAN_CHECKPOINT env var.");
			return;
		}
		if (!new File(finalCheckpointPath).exists()) {
			System.err.println("Error: Checkpoint file not found: " + finalCheckpointPath);
			return;
		}

		// Set environment variable for the generator script
		// Note: This assumes the generator script reads this specific variable.
		// If it doesn't, this won't override the script's internal default.
		// Consider modifying the generator script or passing the path as a command-line
		// argument.
		// For now, we rely on the MARIOGAN_CHECKPOINT convention.
		System.setProperty("MARIOGAN_CHECKPOINT", finalCheckpointPath);
		System.out.println("Using Checkpoint: " + finalCheckpointPath);

		// Create output directory
		File outputDirFile = new File(outputDir);
		if (!outputDirFile.exists()) {
			System.out.println("Creating output directory: " + outputDirFile.getAbsolutePath());
			if (!outputDirFile.mkdirs()) {
				System.err.println("Error: Failed to create output directory.");
				return;
			}
		}

		// Redirect stdout to a file if needed (e.g., for simulation logs)
		// PrintStream originalOut = System.out;
		// PrintStream fileOut = new PrintStream(new FileOutputStream(new
		// File(outputDir, "viewer_log.txt"), true)); // Append mode
		// System.setOut(fileOut);

		// --- Initialize Tile Behaviors (crucial!) ---
		try {
			InputStream is = MarioLevelViewer.class.getResourceAsStream("/ch/idsia/mario/engine/resources/tiles.dat");
			if (is == null) {
				File file = new File("marioaiDagstuhl/src/ch/idsia/mario/engine/resources/tiles.dat"); // Adjust path if
																										// needed
				if (file.exists()) {
					is = new java.io.FileInputStream(file);
				} else {
					throw new IOException("Could not find tiles.dat as resource or file.");
				}
			}
			Level.loadBehaviors(new DataInputStream(is));
			is.close();
		} catch (IOException e) {
			System.err.println("FATAL ERROR: Could not load tile behaviors: " + e.getMessage());
			e.printStackTrace(System.err);
			System.exit(1);
		}

		// --- Level Generation/Loading ---
		// Initialize MarioEvalFunction only if we need to generate from GAN
		MarioEvalFunction eval = null;
		Level level = null;

		if (latentVectorString != null) {
			// Single vector provided
			System.out.println("Generating level from provided latent vector: " + latentVectorString);
			try {
				double[] vector = JsonReader.JsonToDoubleArray(latentVectorString);
				if (latentDim != null && latentDim != vector.length) {
					System.err.println("Error: Provided vector length (" + vector.length
							+ ") doesn't match --latentDim (" + latentDim + ").");
					return;
				}
				// Need MarioEvalFunction to generate the level from vector
				eval = new MarioEvalFunction(false); // Assume visualization off for generation
				level = eval.levelFromLatentVector(vector);
				if (level == null) {
					System.err.println("Error: MarioEvalFunction failed to generate level from vector.");
					return;
				}
			} catch (Exception e) {
				System.err.println("Error parsing or generating level from provided latent vector: " + e.getMessage());
				return;
			}
		} else {
			// Generate random vectors and levels
			if (latentDim == null)
				latentDim = 32; // Default dimension if not provided
			System.out.println("Generating " + numLevels + " levels from random latent vector(s) with dimension "
					+ latentDim + "...");
			eval = new MarioEvalFunction(false); // Initialize for generation
			Random random = new Random();

			// Generate multiple levels if needed
			// For simplicity, this example focuses on processing one level,
			// but the loop below should handle multiple if generation logic is inside the
			// loop.
			// We'll generate the first level here for demonstration.
			if (numLevels > 0) {
				double[] vector = new double[latentDim];
				for (int j = 0; j < latentDim; j++) {
					vector[j] = random.nextDouble() * 2 - 1; // Range [-1, 1]
				}
				level = eval.levelFromLatentVector(vector);
				if (level == null) {
					System.err.println("Error: MarioEvalFunction failed to generate level from random vector.");
					return;
				}
			} else {
				System.err.println("Error: Number of levels to generate must be positive.");
				return;
			}
			// NOTE: This currently only generates and processes the *first* random level.
			// To process multiple random levels, the generation logic needs to be inside
			// the loop below.
		}

		if (level == null) {
			System.err.println("Error: Level object could not be created.");
			return;
		}

		// --- Process Generated/Loaded Level ---
		// This part currently processes only ONE level determined above.
		// To process multiple, you'd loop from 0 to numLevels (if generating random)
		// or handle multiple inputs if loading from files/multiple vectors.
		String baseFilename = String.format("%s_%03d", filenamePrefix, 0); // Using index 0 for the single level
		String fullPathPrefix = new File(outputDir, baseFilename).getAbsolutePath();
		System.out.println("\n--- Processing Level 0 (" + baseFilename + ") ---");

		// Save Image
		if (saveImage) {
			try {
				saveLevelImage(level, fullPathPrefix, clipBuffer);
			} catch (IOException e) {
				System.err.println("Error saving level image: " + e.getMessage());
			}
		}

		// Analyze Features
		if (analyze) {
			System.out.println("Analyzing level features...");
			Map<String, Number> stats = analyzeLevelFeatures(level);
			System.out.println("Level Statistics:");
			stats.forEach((key, value) -> System.out.println("  " + key + ": " + value));
		}

		// Simulate Level
		if (simulate) {
			System.out.println("Simulating level with A* agent..." + (visualizeSim ? " (Visualization ON)" : ""));
			EvaluationInfo simInfo = simulateLevel(level, visualizeSim);
			if (simInfo != null) {
				System.out.println("Simulation Results:");
				System.out.println("  Status: " + (simInfo.marioStatus == Mario.STATUS_WIN ? "WIN" : "LOSE/TIMEOUT"));
				// Try distancePassed
				System.out.println("  Distance Passed: " + simInfo.distancePassed);
				System.out.println("  Time Left: " + simInfo.timeLeft);
				System.out.println("  Mario Mode: " + simInfo.marioMode);
				System.out.println("  Kills Total: " + simInfo.killsTotal);
				// System.out.println(" Memo: " + simInfo.memo);
			} else {
				System.out.println("Simulation failed or produced no results.");
			}
		}

		// Clean up MarioEvalFunction process if it was created
		if (eval != null) {
			eval.exit();
		}

		System.out.println("\n--- Processing Complete ---");
	}
}
