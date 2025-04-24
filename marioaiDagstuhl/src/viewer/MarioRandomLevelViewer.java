package viewer;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.awt.Color;
import java.awt.Image;

import javax.imageio.ImageIO;

import basicMap.Settings;
import ch.idsia.ai.tasks.ProgressTask;
import ch.idsia.mario.engine.LevelRenderer;
import ch.idsia.mario.engine.level.Level;
import ch.idsia.mario.engine.level.LevelParser;
import ch.idsia.tools.CmdLineOptions;
import ch.idsia.tools.EvaluationOptions;
import cmatest.MarioEvalFunction;

/**
 * This file generates several level images by querying the
 * trained GAN with random latent vectors.
 */
public class MarioRandomLevelViewer {

	public static final int BLOCK_SIZE = 16;
	public static final int LEVEL_HEIGHT = 14;
	
	/**
	 * Return an image of the level, excluding 
	 * the background, Mario, and enemy sprites.
	 * @param level
	 * @return
	 */
	public static BufferedImage getLevelImage(Level level, boolean excludeBufferRegion) {
		EvaluationOptions options = new CmdLineOptions(new String[0]);
		ProgressTask task = new ProgressTask(options);
		// Added to change level
        options.setLevel(level);
		task.setOptions(options);

		int relevantWidth = (level.width - (excludeBufferRegion ? 2*LevelParser.BUFFER_WIDTH : 0)) * BLOCK_SIZE;

		BufferedImage image = new BufferedImage(relevantWidth, LEVEL_HEIGHT*BLOCK_SIZE, BufferedImage.TYPE_INT_RGB);
		// Skips buffer zones at start and end of level
		LevelRenderer.renderAreaWhiteBack((Graphics2D) image.getGraphics(), level, 0, 0, excludeBufferRegion ? LevelParser.BUFFER_WIDTH*BLOCK_SIZE : 0, 0, relevantWidth, LEVEL_HEIGHT*BLOCK_SIZE);
		return image;
	}

	/**
	 * Save level as an image
	 * @param level Mario Level
	 * @param name Filename, not including jpg extension
	 * @param clipBuffer Whether to exclude the buffer region we add to all levels
	 * @throws IOException
	 */
	public static void saveLevel(Level level, String name, boolean clipBuffer) throws IOException {
		BufferedImage image = getLevelImage(level, clipBuffer);
		File file = new File(name + ".jpg");
		ImageIO.write(image, "jpg", file);		
		System.out.println("File saved: " + file);
	}

	public static double[] randomUniformDoubleArray(int dim) {
		Random rdm = new Random();
		double[] array = new double[dim];
		for (int i=0; i<dim; i++) {
			array[i] = rdm.nextDouble();
		}
		return array;
	}

	public static double[] randomGaussianDoubleArray(int dim) {
		Random rdm = new Random();
		double[] array = new double[dim];
		for (int i=0; i<dim; i++) {
			array[i] = rdm.nextGaussian();
		}
		return array;
	}
	
	/**
	 * Renders the level map to a BufferedImage manually, 
	 * using Art.getTile without relying on LevelRenderer.
	 * Should work in headless mode if Art.init() succeeded.
	 * @param level Level object
	 * @param excludeBufferRegion Whether to clip buffer zones
	 * @return BufferedImage of the level map
	 */
	public static BufferedImage renderLevelMapHeadless(Level level, boolean excludeBufferRegion) {
		int startX = excludeBufferRegion ? LevelParser.BUFFER_WIDTH : 0;
		int endX = level.width - (excludeBufferRegion ? LevelParser.BUFFER_WIDTH : 0);
		int renderWidth = (endX - startX) * BLOCK_SIZE;
		
		if(renderWidth <= 0) renderWidth = 1; // Ensure width is at least 1
		BufferedImage image = new BufferedImage(renderWidth, LEVEL_HEIGHT * BLOCK_SIZE, BufferedImage.TYPE_INT_RGB);
		Graphics2D g = image.createGraphics();

		// Fill background (e.g., white)
		g.setColor(java.awt.Color.WHITE);
		g.fillRect(0, 0, image.getWidth(), image.getHeight());

		for (int x = startX; x < endX; x++) {
		    for (int y = 0; y < level.height; y++) {
		        int tileByte = level.getBlock(x, y) & 0xFF;
		        // Simple color mapping based on tile byte
		        Color tileColor = new Color(tileByte * 17 % 255, tileByte * 23 % 255, tileByte * 31 % 255);
		        g.setColor(tileColor);
		        g.fillRect((x - startX) * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
		    }
		}
		g.dispose(); // Release graphics resources
		return image;
	}

	/**
	 * Save level as a PNG using manual headless rendering.
	 * @param level Mario Level
	 * @param name Filename, including png extension
	 * @param clipBuffer Whether to exclude the buffer region
	 * @throws IOException
	 */
	public static void saveLevelManually(Level level, String name, boolean clipBuffer) throws IOException {
		BufferedImage image = renderLevelMapHeadless(level, clipBuffer);
		File file = new File(name);
		ImageIO.write(image, "png", file);		
		System.out.println("File saved: " + file);
	}

	public static void main(String[] args) throws IOException {
		Settings.setPythonProgram();

		// This is used because it contains code for communicating with the GAN
		MarioEvalFunction eval = new MarioEvalFunction();

		int nbLevels = 20;
		int dim = 32;
		boolean uniform = false;
		Level level;

		String filenameHead = "";
		if (uniform) {
			filenameHead = "uniform";
		} else {
			filenameHead = "gaussian";
		}
		for (int i=1; i<=nbLevels; i++) {
			if (uniform) {
				level = eval.levelFromLatentVector(randomUniformDoubleArray(dim));
			} else {
				level = eval.levelFromLatentVector(randomGaussianDoubleArray(dim));
			}
			// Replace saveLevel with manual version
			saveLevelManually(level, "randomSamples" + File.separator + filenameHead + "LevelClipped_" + i + ".png", true);
			saveLevelManually(level, "randomSamples" + File.separator + filenameHead+ "LevelFull_" + i + ".png", false);
		}
		eval.exit();
		System.exit(0);
	}
}
