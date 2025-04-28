package cmatest;

import static basicMap.Settings.DEBUG_MSG;

import java.io.IOException;
import java.util.Arrays;

import basicMap.Settings;
import fr.inria.optimization.cmaes.CMAEvolutionStrategy;
import fr.inria.optimization.cmaes.fitness.IObjectiveFunction;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;

public class CMAMarioSolver {
    // Sebastian's Wasserstein GAN expects latent vectors of length 32
    public static final int Z_SIZE = 32; // length of latent space vector
    public static final int EVALS = 1000; // Max evaluations per run

    public static void main(String[] args) throws IOException {
        Settings.setPythonProgram();
        int loops = 100;
        double[][] bestX = new double[loops][32];
        double[] bestY = new double[loops];
        MarioEvalFunction marioEvalFunction = new MarioEvalFunction();
        for (int i = 0; i < loops; i++) {
            System.out.println("Starting CMA-ES Run " + (i + 1) + " of " + loops + "...");
            CMAMarioSolver solver = new CMAMarioSolver(marioEvalFunction, Z_SIZE, EVALS);
            FileWriter write = new FileWriter("timeline" + i + ".txt", true);
            PrintWriter print_line = new PrintWriter(write);
            double[] solution = solver.run(print_line, i + 1, loops);
            print_line.close();
            double currentBestFitness = solver.fitFun.valueOf(MarioEvalFunction.mapArrayToOne(solution));
            System.out.println(
                    "Finished CMA-ES Run " + (i + 1) + " of " + loops + ". Best Fitness Found: " + currentBestFitness);
            bestX[i] = MarioEvalFunction.mapArrayToOne(solution);
            bestY[i] = currentBestFitness;
        }
        marioEvalFunction.exit();
        System.out.println("All " + loops + " CMA-ES runs completed.");
        FileWriter write = new FileWriter("ex_output.txt", true);
        try (PrintWriter print_line = new PrintWriter(write)) {
            for (int i = 0; i < loops; i++) {
                print_line.println(Arrays.toString(bestX[i]));
                // System.out.println(Arrays.toString(bestX[i])); // Commented out for less
                // clutter
            }
            print_line.println(Arrays.toString(bestY));
        }
        System.out.println("Best fitness values across runs: " + Arrays.toString(bestY));
        System.exit(0);
    }

    IObjectiveFunction fitFun;
    int nDim;
    CMAEvolutionStrategy cma;
    int maxEvals; // Store maxEvals

    public CMAMarioSolver(IObjectiveFunction fitFun, int nDim, int maxEvals) {
        this.fitFun = fitFun;
        this.nDim = nDim;
        this.maxEvals = maxEvals; // Store maxEvals
        cma = new CMAEvolutionStrategy();
        cma.readProperties(); // read options, see file CMAEvolutionStrategy.properties
        cma.setDimension(nDim); // overwrite some loaded properties
        cma.setInitialX(-1, 1); // set initial seach point xmean coordinate-wise uniform between l and u,
                                // dimension needs to have been set before
        cma.setInitialStandardDeviation(1 / Math.sqrt(nDim)); // also a mandatory setting
        cma.options.stopFitness = -1e6; // 1e-14; // optional setting
        // cma.options.stopMaxIter = 100;
        cma.options.stopMaxFunEvals = maxEvals;
        // System.out.println("Diagonal: " + cma.options.diagonalCovarianceMatrix); //
        // Removed this line
    }

    public void setDim(int n) {
        cma.setDimension(n);
    }

    /*
     * public void setInitialX(double x) {
     * cma.setInitialX(x);
     * }
     */

    public void setObjective(IObjectiveFunction fitFun) {
        this.fitFun = fitFun;
    }

    public void setMaxEvals(int n) {
        cma.options.stopMaxFunEvals = n;
        this.maxEvals = n; // Update stored maxEvals
    }

    public double[] run(PrintWriter print_line, int currentRun, int totalRuns) {

        // new a CMA-ES and set some initial values

        // initialize cma and get fitness array to fill in later
        double[] fitness = cma.init(); // new double[cma.parameters.getPopulationSize()];

        // initial output to files
        cma.writeToDefaultFilesHeaders(0); // 0 == overwrites old files

        // iteration loop
        while (cma.stopConditions.getNumber() == 0) {

            // Get evaluation count at the start of the generation
            long evalsAtStartOfGen = cma.getCountEval();

            // --- core iteration step ---
            double[][] pop = cma.samplePopulation(); // get a new population of solutions
            for (int i = 0; i < pop.length; ++i) { // for each candidate solution i
                // a simple way to handle constraints that define a convex feasible domain
                // (like box constraints, i.e. variable boundaries) via "blind re-sampling"
                // assumes that the feasible domain is convex, the optimum is
                while (!fitFun.isFeasible(pop[i])) { // not located on (or very close to) the domain boundary,
                    // System.out.println(DEBUG_MSG + "Not in feasible domain. Will resample
                    // once."); // Removed this line
                    pop[i] = cma.resampleSingle(i); // initialX is feasible and initialStandardDeviations are
                    // sufficiently small to prevent quasi-infinite looping here
                    // compute fitness/objective value
                }
                fitness[i] = fitFun.valueOf(pop[i]); // fitfun.valueOf() is to be minimized
                // System.out.println(fitness[i]); // Removed this line
                print_line.println(Arrays.toString(pop[i]) + " : " + fitness[i]); // Still write to timeline file

                // --- Update Progress Bar After Each Evaluation ---
                // Calculate count based on start of generation + loop index
                long currentEvalForProgress = evalsAtStartOfGen + i + 1;
                // Use maxEvals field which should be same as cma.options.stopMaxFunEvals
                int maxEval = this.maxEvals;
                // Calculate progress based on the calculated count
                double progress = (maxEval > 0) ? (double) currentEvalForProgress / maxEval : 0;
                int barWidth = 50; // Width of the progress bar
                int filledWidth = (int) (barWidth * progress);
                // Ensure filledWidth doesn't exceed barWidth due to potential floating point
                // inaccuracies
                filledWidth = Math.min(barWidth, filledWidth);

                String bar = "";
                if (filledWidth > 0) {
                    bar = String.join("", java.util.Collections.nCopies(filledWidth - 1, "="));
                    if (filledWidth <= barWidth) {
                        bar += ">";
                    }
                }
                String paddedBar = String.format("[%-" + barWidth + "s]", bar);

                // Print progress bar to System.err to avoid interfering with Python script
                // stdout
                // Include Run X/Y
                // Use simpler printf with \r - overwriting might not work reliably in all
                // terminals.
                System.err.printf("\rRun %d/%d | Evaluations: %d/%d %s %.1f%%",
                        currentRun, totalRuns, currentEvalForProgress, maxEval, paddedBar, progress * 100);
                System.err.flush(); // Ensure the error stream output is written immediately
                // --- End Inner Progress Bar Update ---
            }
            cma.updateDistribution(fitness); // pass fitness array to update search distribution
            // --- end core iteration step ---

            // --- Progress Bar --- (REMOVED - Now updated inside inner loop)
            // System.out.flush(); // Attempt to flush stdout before writing progress to
            // stderr
            // long currentEval = cma.getCountEval();
            // // Use maxEvals field which should be same as cma.options.stopMaxFunEvals
            // int maxEval = this.maxEvals;
            // double progress = (maxEval > 0) ? (double) currentEval / maxEval : 0;
            // int barWidth = 50; // Width of the progress bar
            // int filledWidth = (int) (barWidth * progress);
            // // Ensure filledWidth doesn't exceed barWidth due to potential floating point
            // // inaccuracies
            // filledWidth = Math.min(barWidth, filledWidth);

            // String bar = "";
            // if (filledWidth > 0) {
            // bar = String.join("", java.util.Collections.nCopies(filledWidth - 1, "="));
            // if (filledWidth <= barWidth) {
            // bar += ">";
            // }
            // }
            // String paddedBar = String.format("[%-" + barWidth + "s]", bar);

            // // Print progress bar to System.err to avoid interfering with Python script
            // stdout
            // // Include Run X/Y
            // // Use simpler printf with \r - overwriting might not work reliably in all
            // // terminals.
            // System.err.printf("\rRun %d/%d | Evaluations: %d/%d %s %.1f%%",
            // currentRun, totalRuns, currentEval, maxEval, paddedBar, progress * 100);
            // System.err.flush(); // Ensure the error stream output is written immediately
            // --- End Progress Bar ---

            // Removed default CMA-ES console output
            // cma.writeToDefaultFiles(); // Keep writing data files if needed
            // int outmod = 150;
            // if (cma.getCountIter() % (15 * outmod) == 1) {
            // // cma.printlnAnnotation(); // might write file as well
            // }
            // if (cma.getCountIter() % outmod == 1) {
            // // cma.println();
            // }
        }
        System.err.println(); // Add a newline to System.err after the progress bar finishes

        // evaluate mean value as it is the best estimator for the optimum
        // cma.setFitnessOfMeanX(fitFun.valueOf(cma.getMeanX())); // updates the best
        // ever solution

        // final output
        cma.writeToDefaultFiles(1); // Keep writing final data files if needed
        // cma.println(); // Removed default final prints for less clutter
        cma.println("Terminated due to:"); // Keep termination info
        for (String s : cma.stopConditions.getMessages())
            cma.println("  " + s);
        cma.println("Best function value " + cma.getBestFunctionValue()
                + " at evaluation " + cma.getBestEvaluationNumber());

        // System.out.println("Best solution is: " + Arrays.toString(cma.getBestX()));
        // we might return cma.getBestSolution() or cma.getBestX()
        // return cma.getBestX();
        cma.setFitnessOfMeanX(fitFun.valueOf(cma.getMeanX())); // updates the best ever solution
        return cma.getBestX();
        // return cma.getBestRecentX();

    }

}
