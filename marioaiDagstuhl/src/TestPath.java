import java.io.File;

public class TestPath {
    public static void main(String[] args) {
        File pythonExec = new File("/home/Angan/MarioGAN/venv/bin/python3");
        System.out.println("Checking path: " + pythonExec.getAbsolutePath());
        System.out.println("Exists? " + pythonExec.exists());
        System.out.println("Is File? " + pythonExec.isFile());
        System.out.println("Can Execute? " + pythonExec.canExecute());

        File binDir = new File("/home/Angan/MarioGAN/venv/bin");
        System.out.println("\nListing directory: " + binDir.getAbsolutePath());
        if (binDir.exists() && binDir.isDirectory()) {
            for (File f : binDir.listFiles()) {
                System.out.println("  - " + f.getName());
            }
        } else {
            System.out.println("Bin directory does not exist or is not a directory.");
        }
    }
} 