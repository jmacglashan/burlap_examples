package edu.brown.cs.burlap.examples;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learnfromdemo.RewardValueProjection;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRL;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest;
import burlap.behavior.singleagent.learnfromdemo.mlirl.commonrfs.LinearStateDifferentiableRF;
import burlap.behavior.singleagent.learnfromdemo.mlirl.differentiableplanners.DifferentiableSparseSampling;
import burlap.behavior.singleagent.vfa.StateToFeatureVectorGenerator;
import burlap.behavior.valuefunction.QFunction;
import burlap.debugtools.RandomFactory;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.auxiliary.common.NullTermination;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.GroundedProp;
import burlap.oomdp.core.PropositionalFunction;
import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.NullRewardFunction;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.oomdp.visualizer.Visualizer;

import java.util.List;

/**
 * Example code for performing IRL. Choose the steps to perform in the main method by one of the three primary methods.
 * Create your demonstrations using the {@link #launchExplorer()} method. See its documentation
 * which describes how to use the standard BURLAP shell to record and save your demonstrations to disk.
 * If you wish to verify which demonstrations were saved to disk, use the {@link #launchSavedEpisodeSequenceVis(String)}
 * method, giving it the path to where you saved the data. Then, use the {@link #runIRL(String)} method giving it
 * the same path to the saved demonstrations to actually run IRL. It will then visualize the learned reward function
 * @author James MacGlashan.
 */
public class IRLExample {

	GridWorldDomain gwd;
	Domain domain;
	StateGenerator sg;
	Visualizer v;

	public IRLExample(){

		this.gwd = new GridWorldDomain(5 ,5);
		this.gwd.setNumberOfLocationTypes(5);
		gwd.makeEmptyMap();
		this.domain = gwd.generateDomain();
		State bs = this.basicState();
		this.sg = new LeftSideGen(5, bs);
		this.v = GridWorldVisualizer.getVisualizer(this.gwd.getMap());

	}

	/**
	 * Creates a visual explorer that you can use to to record trajectories. Use the "`" key to reset to a random initial state
	 * Use the wasd keys to move north south, east, and west, respectively. To enable recording,
	 * first open up the shell and type: "rec -b" (you only need to type this one). Then you can move in the explorer as normal.
	 * Each demonstration begins after an environment reset.
	 * After each demonstration that you want to keep, go back to the shell and type "rec -r"
	 * If you reset the environment before you type that,
	 * the episode will be discarded. To temporarily view the episodes you've created, in the shell type "episode -v". To actually record your
	 * episodes to file, type "rec -w path/to/save/directory base_file_name" For example "rec -w irl_demos demo"
	 * A recommendation for examples is to record two demonstrations that both go to the pink cell while avoiding blue ones
	 * and do so from two different start locations on the left (if you keep resetting the environment, it will change where the agent starts).
	 */
	public void launchExplorer(){
		SimulatedEnvironment env = new SimulatedEnvironment(this.domain, new NullRewardFunction(), new NullTermination(), this.sg);
		VisualExplorer exp = new VisualExplorer(this.domain, env, this.v, 800, 800);
		exp.addKeyAction("w", GridWorldDomain.ACTIONNORTH);
		exp.addKeyAction("s", GridWorldDomain.ACTIONSOUTH);
		exp.addKeyAction("d", GridWorldDomain.ACTIONEAST);
		exp.addKeyAction("a", GridWorldDomain.ACTIONWEST);

		//exp.enableEpisodeRecording("r", "f", "irlDemo");

		exp.initGUI();
	}


	/**
	 * Launch a episode sequence visualizer to display the saved trajectories in the folder "irlDemo"
	 */
	public void launchSavedEpisodeSequenceVis(String pathToEpisodes){

		EpisodeSequenceVisualizer evis = new EpisodeSequenceVisualizer(this.v, this.domain, pathToEpisodes);

	}

	/**
	 * Runs MLIRL on the trajectories stored in the "irlDemo" directory and then visualizes the learned reward function.
	 */
	public void runIRL(String pathToEpisodes){

		//create reward function features to use
		LocationFV fvg = new LocationFV(this.domain, 5);

		//create a reward function that is linear with respect to those features and has small random
		//parameter values to start
		LinearStateDifferentiableRF rf = new LinearStateDifferentiableRF(fvg, 5);
		for(int i = 0; i < rf.numParameters(); i++){
			rf.setParameter(i, RandomFactory.getMapped(0).nextDouble()*0.2 - 0.1);
		}

		//load our saved demonstrations from disk
		List<EpisodeAnalysis> episodes = EpisodeAnalysis.parseFilesIntoEAList(pathToEpisodes, domain);

		//use either DifferentiableVI or DifferentiableSparseSampling for planning. The latter enables receding horizon IRL,
		//but you will probably want to use a fairly large horizon for this kind of reward function.
		double beta = 10.;
		//DifferentiableVI dplanner = new DifferentiableVI(this.domain, rf, new NullTermination(), 0.99, 8, new SimpleHashableStateFactory(), 0.01, 100);
		DifferentiableSparseSampling dplanner = new DifferentiableSparseSampling(this.domain, rf, new NullTermination(), 0.99, new SimpleHashableStateFactory(), 10, -1, beta);


		dplanner.toggleDebugPrinting(false);

		//define the IRL problem
		MLIRLRequest request = new MLIRLRequest(domain, dplanner, episodes, rf);
		request.setBoltzmannBeta(beta);

		//run MLIRL on it
		MLIRL irl = new MLIRL(request, 0.1, 0.1, 10);
		irl.performIRL();


		//get all states in the domain so we can visualize the learned reward function for them
		List<State> allStates = StateReachability.getReachableStates(basicState(), (SADomain) this.domain, new SimpleHashableStateFactory());

		//get a standard grid world value function visualizer, but give it StateRewardFunctionValue which returns the
		//reward value received upon reaching each state which will thereby let us render the reward function that is
		//learned rather than the value function for it.
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
				allStates,
				new RewardValueProjection(rf),
				new GreedyQPolicy((QFunction)request.getPlanner()));


		gui.initGUI();


	}


	/**
	 * Creates a grid world state with the agent in (0,0) and various different grid cell types scattered about.
	 * @return a grid world state with the agent in (0,0) and various different grid cell types scattered about.
	 */
	protected State basicState(){

		State s = GridWorldDomain.getOneAgentNLocationState(this.domain, 9);
		GridWorldDomain.setAgent(s, 0, 0);

		//goals
		GridWorldDomain.setLocation(s, 0, 0, 0, 1);
		GridWorldDomain.setLocation(s, 1, 0, 4, 2);
		GridWorldDomain.setLocation(s, 2, 4, 4, 3);
		GridWorldDomain.setLocation(s, 3, 4, 0, 4);

		GridWorldDomain.setLocation(s, 4, 1, 0, 0);
		GridWorldDomain.setLocation(s, 5, 1, 2, 0);
		GridWorldDomain.setLocation(s, 6, 1, 4, 0);

		GridWorldDomain.setLocation(s, 7, 3, 1, 0);
		GridWorldDomain.setLocation(s, 8, 3, 3, 0);

		return s;
	}

	/**
	 * State generator that produces initial agent states somewhere on the left side of the grid.
	 */
	public static class LeftSideGen implements StateGenerator{


		protected int height;
		protected State sourceState;


		public LeftSideGen(int height, State sourceState){
			this.setParams(height, sourceState);
		}

		public void setParams(int height, State sourceState){
			this.height = height;
			this.sourceState = sourceState;
		}



		public State generateState() {

			State s = this.sourceState.copy();

			int h = RandomFactory.getDefault().nextInt(this.height);
			GridWorldDomain.setAgent(s, 0, h);

			return s;
		}
	}

	/**
	 * A state feature vector generator that create a binary feature vector where each element
	 * indicates whether the agent is in a cell of of a different type. All zeros indicates
	 * that the agent is in an empty cell.
	 */
	public static class LocationFV implements StateToFeatureVectorGenerator {

		protected int numLocations;
		PropositionalFunction inLocaitonPF;


		public LocationFV(Domain domain, int numLocations){
			this.numLocations = numLocations;
			this.inLocaitonPF = domain.getPropFunction(GridWorldDomain.PFATLOCATION);
		}



		public double[] generateFeatureVectorFrom(State s) {

			double [] fv = new double[this.numLocations];

			int aL = this.getActiveLocationVal(s);
			if(aL != -1){
				fv[aL] = 1.;
			}

			return fv;
		}


		protected int getActiveLocationVal(State s){

			List<GroundedProp> gps = this.inLocaitonPF.getAllGroundedPropsForState(s);
			for(GroundedProp gp : gps){
				if(gp.isTrue(s)){
					ObjectInstance l = s.getObject(gp.params[1]);
					int lt = l.getIntValForAttribute(GridWorldDomain.ATTLOCTYPE);
					return lt;
				}
			}

			return -1;
		}
	}

	public static void main(String[] args) {

		IRLExample ex = new IRLExample();

		//only have one of the below uncommented

		ex.launchExplorer(); //choose this to record demonstrations
		//ex.launchSavedEpisodeSequenceVis("irl_demos"); //choose this review the demonstrations that you've recorded
		//ex.runIRL("irl_demos"); //choose this to run MLIRL on the demonstrations and visualize the learned reward function and policy


	}

}
