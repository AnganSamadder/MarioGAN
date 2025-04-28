package ch.idsia.mario.engine;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.GraphicsConfiguration;
import java.awt.Image;
import java.awt.Transparency;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.File;
import java.util.Iterator;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

public class Art {
    public static Image[][] mario;
    public static Image[][] smallMario;
    public static Image[][] fireMario;
    public static Image[][] enemies;
    public static Image[][] items;
    public static Image[][] level;
    public static Image[][] particles;
    public static Image[][] font;
    public static Image[][] bg;
    public static Image[][] map;
    public static Image[][] endScene;
    public static Image[][] gameOver;
    public static Image logo;
    public static Image titleScreen;

    public static void init(GraphicsConfiguration gc) {
        try {
            // System.out.println("Attempting to initialize Art...");
            // Try loading one essential image to see if resources are found
            // If this fails, the others likely will too.
            Image testLoad = getImage(gc, "mapsheet.png");
            if (testLoad == null) {
                throw new IOException("Failed to load essential image resource mapsheet.png");
            }

            mario = cutImage(gc, "mariosheet.png", 32, 32);
            smallMario = cutImage(gc, "smallmariosheet.png", 16, 16);
            fireMario = cutImage(gc, "firemariosheet.png", 32, 32);
            enemies = cutImage(gc, "enemysheet.png", 16, 32);
            items = cutImage(gc, "itemsheet.png", 16, 16);
            level = cutImage(gc, "mapsheet.png", 16, 16);
            map = cutImage(gc, "worldmap.png", 16, 16);
            particles = cutImage(gc, "particlesheet.png", 8, 8);
            bg = cutImage(gc, "bgsheet.png", 32, 32);
            logo = getImage(gc, "logo.gif");
            titleScreen = getImage(gc, "title.gif");
            font = cutImage(gc, "font.gif", 8, 8);
            endScene = cutImage(gc, "endscene.gif", 96, 96);
            gameOver = cutImage(gc, "gameovergost.gif", 96, 64);
            // System.out.println("Art initialized successfully.");
        } catch (Exception e) {
            System.err.println("Error initializing Art resources:");
            e.printStackTrace();
            // Optionally re-throw or handle more gracefully
        }
    }

    private static Image getImage(GraphicsConfiguration gc, String imageName) throws IOException {
        BufferedImage source = null;
        // Attempt 1: Load via classpath relative to Art class
        try {
            // Use leading slash for path relative to classpath root if img is top-level in src or bin
            // String resourcePath = "/img/" + imageName; 
            // Or try relative path if img is in the same package structure
            // String resourcePath = "img/" + imageName;
            // Let's try the simple name first, assuming it's findable in the classpath
            java.io.InputStream stream = Art.class.getResourceAsStream(imageName);
            if (stream != null) {
                 // System.out.println("Loading " + imageName + " via getResourceAsStream...");
                 source = ImageIO.read(stream);
            } else {
                 // System.out.println("getResourceAsStream failed for " + imageName);
            }
        } catch (Exception e) {
            System.err.println("Exception loading " + imageName + " via getResourceAsStream:");
            e.printStackTrace(); // Log error but continue to fallback
            source = null;
        }

        // Attempt 2: Fallback to file path relative to CWD (less reliable, but was original fallback)
        if (source == null) {
            String filePath = "marioaiDagstuhl/img/" + imageName; // Relative path from project root
            // System.out.println("Attempting fallback load from file path: " + filePath);
            File file = new File(filePath);
            if (!file.exists()) {
                // System.out.println("File does not exist: " + file.getAbsolutePath());
            } else if (!file.canRead()) {
                // System.out.println("File cannot be read: " + file.getAbsolutePath());
            } else {
                try {
                    source = ImageIO.read(file);
                    // System.out.println("Successfully loaded from file path.");
                } catch (IOException e) {
                    System.err.println("IOException loading " + imageName + " from file " + file.getAbsolutePath() + ":");
                    e.printStackTrace();
                    source = null;
                }
            }
        }

        // If still null after all attempts, fail definitively
        if (source == null) {
            throw new IOException("Failed to load image resource '" + imageName + "' using both getResourceAsStream and file path (marioaiDagstuhl/img/" + imageName + ")");
        }

        // Create compatible image, handling null GraphicsConfiguration for headless mode
        Image image;
        if (gc != null) {
            image = gc.createCompatibleImage(source.getWidth(), source.getHeight(), Transparency.BITMASK);
        } else {
            // Create a standard BufferedImage if no GraphicsConfiguration is available
            image = new BufferedImage(source.getWidth(), source.getHeight(), BufferedImage.TYPE_INT_ARGB);
        }
        
        Graphics2D g = (Graphics2D) image.getGraphics();
        g.setComposite(AlphaComposite.Src); // Ensure transparency is handled correctly
        g.drawImage(source, 0, 0, null);
        g.dispose();
        return image;
    }

    private static Image[][] cutImage(GraphicsConfiguration gc, String imageName, int xSize, int ySize)
            throws IOException {
        Image source = getImage(gc, imageName);
        // Handle potential null from failed getImage call within init's try-catch
        if (source == null) {
            System.err.println("Cannot cutImage for " + imageName + " because source image failed to load.");
            // Return an empty array or handle error appropriately
            return new Image[0][0]; // Example: return empty array
        }
        Image[][] images = new Image[source.getWidth(null) / xSize][source.getHeight(null) / ySize];
        for (int x = 0; x < source.getWidth(null) / xSize; x++) {
            for (int y = 0; y < source.getHeight(null) / ySize; y++) {
                // Use same logic for creating image as in getImage
                Image tileImage;
                if (gc != null) {
                    tileImage = gc.createCompatibleImage(xSize, ySize, Transparency.BITMASK);
                } else {
                    tileImage = new BufferedImage(xSize, ySize, BufferedImage.TYPE_INT_ARGB);
                }

                Graphics2D g = (Graphics2D) tileImage.getGraphics();
                g.setComposite(AlphaComposite.Src);
                g.drawImage(source, -x * xSize, -y * ySize, null);
                g.dispose();
                images[x][y] = tileImage;
            }
        }

        return images;
    }

}