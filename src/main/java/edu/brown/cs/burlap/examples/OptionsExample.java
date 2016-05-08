package edu.brown.cs.burlap.examples;

import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.options.DeterministicTerminationOption;
import burlap.behavior.singleagent.options.Option;
import burlap.behavior.singleagent.planning.deterministic.DDPlannerPolicy;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.valuefunction.ValueFunctionInitialization;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.core.Domain;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.oo.state.OOVariableKey;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.RewardFunction;
import burlap.mdp.singleagent.common.UniformCostRF;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.statehashing.SimpleHashableStateFactory;
import burlap.mdp.visualizer.Visualizer;

import java.util.ArrayList;
import java.util.List;

/**
 * Example code showing how options can be constructed and used. The policy for these options were created using
 * a planning algorithm. In general, if you're doing learning, you shouldn't have access to the transition function
 * to do the planning, but here we're just using planning as a means to easily generate the policy to use for purposes
 * of illustration.
 *
 * You can use the {@link #testOptions()} method to see the outcome of applying an option and use the
 * {@link #optionComparison()} method to see how Q-learning w/ options compares to vanilla Q-learning.
 * Note that this method shows the results of providing Q-learning different sets of options.
 * For example, when you provide all options
 * Q-learning will actually do worse than vanilla Q-learning. But if you remove bad options, then can see
 * see improved performance. This effect demonstrates how options in general need to be selected carefully to avoid
 * making the problem harder. Other things you might want to try is using special value function initialization
 * so that good options have a higher initial Q-value than primitives, and bad options have lower values.
 * For that, create a {@link burlap.behavior.valuefunction.ValueFunctionInitialization} object and hand it
 * to your {@link QLearning} instances with the {@link QLearning#setQInitFunction(ValueFunctionInitialization)} method.
 * @author James MacGlashan.
 */
public class OptionsExample {

	public static void testOptions(){

		GridWorldDomain gwd = new GridWorldDomain(11, 11);
		gwd.setMapToFourRooms();
		Domain domain = gwd.generateDomain();

		Option swToNorth = createRoomOption("swToNorth", domain, 1, 5, 0, 0, 4, 4);
		Option swToEast = createRoomOption("swToEast", domain, 5, 1, 0, 0, 4, 4);

		Option seToWest = createRoomOption("seToWest", domain, 5, 1, 6, 0, 10, 3);
		Option seToNorth = createRoomOption("seToNorth", domain, 8, 4, 6, 0, 10, 3);

		Option neToSouth = createRoomOption("neToSouth", domain, 8, 4, 6, 5, 10, 10);
		Option neToWest = createRoomOption("neToWest", domain, 5, 8, 6, 5, 10, 10);

		Option nwToEast = createRoomOption("nwToEast", domain, 5, 8, 0, 6, 4, 10);
		Option nwToSouth = createRoomOption("nwToSouth", domain, 1, 5, 0, 6, 4, 10);

		List<EpisodeAnalysis> episodes = new ArrayList<EpisodeAnalysis>();

		episodes.add(optionExecuteResult(swToNorth, new GridWorldState(0, 0)));
		episodes.add(optionExecuteResult(swToEast, new GridWorldState(0, 0)));

		episodes.add(optionExecuteResult(seToWest, new GridWorldState(10, 0)));
		episodes.add(optionExecuteResult(seToNorth, new GridWorldState(10, 0)));

		episodes.add(optionExecuteResult(neToSouth, new GridWorldState(10, 10)));
		episodes.add(optionExecuteResult(neToWest, new GridWorldState(10, 10)));

		episodes.add(optionExecuteResult(nwToEast, new GridWorldState(0, 10)));
		episodes.add(optionExecuteResult(nwToSouth, new GridWorldState(0, 10)));


		Visualizer v = GridWorldVisualizer.getVisualizer(gwd.getMap());
		EpisodeSequenceVisualizer evis = new EpisodeSequenceVisualizer(v, domain, episodes);


	}

	public static EpisodeAnalysis optionExecuteResult(Option o, State s){
		o.getAssociatedGroundedAction().executeIn(s);
		return o.getLastExecutionResults();
	}

	public static void optionComparison(){

		//set up four rooms learning problem with the goal in the most north-east cell (10,10) and initial
		//state in the most south-west cell (0,0).
		GridWorldDomain gwd = new GridWorldDomain(11, 11);
		gwd.setMapToFourRooms();
		//gwd.setProbSucceedTransitionDynamics(0.8);
		final Domain domain = gwd.generateDomain();
		State s = new GridWorldState(0, 0);

		RewardFunction rf = new UniformCostRF();
		TerminalFunction tf = new GridWorldTerminalFunction(10, 10);
		SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, s);

		final Option swn = createRoomOption("swToNorth", domain, 1, 5, 0, 0, 4, 4);
		final Option swe = createRoomOption("swToEast", domain, 5, 1, 0, 0, 4, 4);

		final Option sew = createRoomOption("seToWest", domain, 5, 1, 6, 0, 10, 3);
		final Option sen = createRoomOption("seToNorth", domain, 8, 4, 6, 0, 10, 3);

		final Option nes = createRoomOption("neToSouth", domain, 8, 4, 6, 5, 10, 10);
		final Option newe = createRoomOption("neToWest", domain, 5, 8, 6, 5, 10, 10);

		final Option nwe = createRoomOption("nwToEast", domain, 5, 8, 0, 6, 4, 10);
		final Option nws = createRoomOption("nwToSouth", domain, 1, 5, 0, 6, 4, 10);

		final double qinit = 0;
		final double lr = 1.;

		LearningAgentFactory noOptions = new LearningAgentFactory() {

			public String getAgentName() {
				return "Vanilla Q-Learning";
			}


			public LearningAgent generateAgent() {
				return new QLearning(domain, 0.99, new SimpleHashableStateFactory(), qinit, lr);
			}
		};


		LearningAgentFactory allOptons = new LearningAgentFactory() {

			public String getAgentName() {
				return "All Options";
			}


			public LearningAgent generateAgent() {
				QLearning ql = new QLearning(domain, 0.99, new SimpleHashableStateFactory(), qinit, lr);
				ql.addNonDomainReferencedAction(swn);
				ql.addNonDomainReferencedAction(swe);

				ql.addNonDomainReferencedAction(sew);
				ql.addNonDomainReferencedAction(sen);

				ql.addNonDomainReferencedAction(nes);
				ql.addNonDomainReferencedAction(newe);

				ql.addNonDomainReferencedAction(nwe);
				ql.addNonDomainReferencedAction(nws);

				return ql;
			}
		};

		LearningAgentFactory usefulOptions = new LearningAgentFactory() {

			public String getAgentName() {
				return "No Goal-room Options";
			}


			public LearningAgent generateAgent() {
				QLearning ql = new QLearning(domain, 0.99, new SimpleHashableStateFactory(), qinit, lr);
				ql.addNonDomainReferencedAction(swn);
				ql.addNonDomainReferencedAction(swe);

				ql.addNonDomainReferencedAction(sew);
				ql.addNonDomainReferencedAction(sen);


				ql.addNonDomainReferencedAction(nwe);
				ql.addNonDomainReferencedAction(nws);

				return ql;
			}
		};

		LearningAgentFactory idealOptions = new LearningAgentFactory() {

			public String getAgentName() {
				return "One Way Directed Options";
			}


			public LearningAgent generateAgent() {
				QLearning ql = new QLearning(domain, 0.99, new SimpleHashableStateFactory(), qinit, lr);
				ql.addNonDomainReferencedAction(swn);

				ql.addNonDomainReferencedAction(sen);

				ql.addNonDomainReferencedAction(nwe);

				return ql;
			}
		};


		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env, 10, 100, noOptions, allOptons, usefulOptions, idealOptions);
		exp.setUpPlottingConfiguration(500, 300, 2, 800, TrialMode.MOSTRECENTANDAVERAGE, PerformanceMetric.CUMULATIVESTEPSPEREPISODE);

		exp.startExperiment();


	}


	/**
	 * Method for creating a four rooms option
	 * @param optionName the name of the option
	 * @param domain the burlap domain to which it is associated
	 * @param doorx the x position of the doorway the option will go to
	 * @param doory the y position of the doorway the option will go to
	 * @param minX the minimum x value of the room
	 * @param minY the minimum y value of the room
	 * @param maxX the maximum x value of the room
	 * @param maxY the maximum y value of the room
	 * @return an option take the agent anywhere within the specified room to the designated doorway
	 */
	public static Option createRoomOption(String optionName, final Domain domain, final int doorx, final int doory, final int minX, final int minY, final int maxX, final int maxY){


		//initiation conditions for options are anywhere in the defined room region
		final StateConditionTest initiationConditions = new StateConditionTest() {

			public boolean satisfies(State s) {
				int x = (Integer)s.get(new OOVariableKey(GridWorldDomain.CLASS_AGENT, GridWorldDomain.VAR_X));
				int y = (Integer)s.get(new OOVariableKey(GridWorldDomain.CLASS_AGENT, GridWorldDomain.VAR_Y));

				return x >= minX && x <= maxX && y>= minY && y <= maxY;
			}
		};

		//termination conditions are any states not in the initiation set
		StateConditionTest terminationConditions = new StateConditionTest() {

			public boolean satisfies(State s) {
				return !initiationConditions.satisfies(s);
			}
		};

		//a goal condition so we can use a planning algorithm to generate the option policy
		StateConditionTest goalCondition = new StateConditionTest() {

			public boolean satisfies(State s) {
				int x = (Integer)s.get(new OOVariableKey(GridWorldDomain.CLASS_AGENT, GridWorldDomain.VAR_X));
				int y = (Integer)s.get(new OOVariableKey(GridWorldDomain.CLASS_AGENT, GridWorldDomain.VAR_Y));
				return x == doorx && y == doory;
			}
		};


		//for simplicity of the demonstration of using options, I will compute an option's policy using a planning algorithm
		//if you're trying to solve an RL problem, in practice you wouldn't be able to do this since
		//you assume that the transition dynamics are unknown to the agent.
		//BFS is sufficient for generating the policy to navigate to a hallway when grid world is deterministic
		BFS bfs = new BFS(domain, goalCondition, new SimpleHashableStateFactory());
		bfs.toggleDebugPrinting(false);

		//using a dynamic deterministic planner policy allows BFS to be lazily called to compute the policy of each state in the room
		//BFS will also automatically cache the solution for states it's already seen.
		Policy optionPolicy = new DDPlannerPolicy(bfs);

		//now that we have the parts of our option, instantiate it
		DeterministicTerminationOption option = new DeterministicTerminationOption(optionName, optionPolicy, initiationConditions, terminationConditions);

		return option;
	}




	public static void main(String[] args) {

		//testOptions();
		optionComparison();
	}

}
