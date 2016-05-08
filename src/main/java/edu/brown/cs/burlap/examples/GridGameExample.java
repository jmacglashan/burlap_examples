package edu.brown.cs.burlap.examples;

import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.stochasticgames.GameAnalysis;
import burlap.behavior.stochasticgames.PolicyFromJointPolicy;
import burlap.behavior.stochasticgames.agents.interfacing.singleagent.LearningAgentToSGAgentInterface;
import burlap.behavior.stochasticgames.agents.madp.MultiAgentDPPlanningAgent;
import burlap.behavior.stochasticgames.agents.maql.MultiAgentQLearning;
import burlap.behavior.stochasticgames.auxiliary.GameSequenceVisualizer;
import burlap.behavior.stochasticgames.madynamicprogramming.backupOperators.CoCoQ;
import burlap.behavior.stochasticgames.madynamicprogramming.backupOperators.CorrelatedQ;
import burlap.behavior.stochasticgames.madynamicprogramming.dpplanners.MAValueIteration;
import burlap.behavior.stochasticgames.madynamicprogramming.policies.ECorrelatedQJointPolicy;
import burlap.behavior.stochasticgames.madynamicprogramming.policies.EGreedyMaxWellfare;
import burlap.behavior.stochasticgames.solvers.CorrelatedEquilibriumSolver;
import burlap.debugtools.DPrint;
import burlap.domain.stochasticgames.gridgame.GGVisualizer;
import burlap.domain.stochasticgames.gridgame.GridGame;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.statehashing.HashableStateFactory;
import burlap.mdp.statehashing.SimpleHashableStateFactory;
import burlap.mdp.stochasticgames.JointReward;
import burlap.mdp.stochasticgames.SGAgentType;
import burlap.mdp.stochasticgames.World;
import burlap.mdp.stochasticgames.oo.OOSGDomain;
import burlap.mdp.visualizer.Visualizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Example code showing the usage of CoCo-Q and Correlated-Q operators with planning (VI) and learning (Q-learning),
 * and using single agent learning algorithms on two-player Grid Games (a multi-agent stochastic game).
 * From main, comment/uncomment the example method you want to run.
 * @author James MacGlashan.
 */
public class GridGameExample {

	public static void VICoCoTest(){

		//grid game domain
		GridGame gridGame = new GridGame();
		final OOSGDomain domain = gridGame.generateDomain();

		final HashableStateFactory hashingFactory = new SimpleHashableStateFactory();

		//run the grid game version of prisoner's dilemma
		final State s = GridGame.getPrisonersDilemmaInitialState();

		//define joint reward function and termination conditions for this game
		JointReward rf = new GridGame.GGJointRewardFunction(domain, -1, 100, false);
		TerminalFunction tf = new GridGame.GGTerminalFunction(domain);

		//both agents are standard: access to all actions
		SGAgentType at = GridGame.getStandardGridGameAgentType(domain);

		//create our multi-agent planner
		MAValueIteration vi = new MAValueIteration(domain, rf, tf, 0.99, hashingFactory, 0., new CoCoQ(), 0.00015, 50);

		//instantiate a world in which our agents will play
		World w = new World(domain, rf, tf, s);


		//create a greedy joint policy from our planner's Q-values
		EGreedyMaxWellfare jp0 = new EGreedyMaxWellfare(0.);
		jp0.setBreakTiesRandomly(false); //don't break ties randomly

		//create agents that follows their end of the computed the joint policy
		MultiAgentDPPlanningAgent a0 = new MultiAgentDPPlanningAgent(domain, vi, new PolicyFromJointPolicy(jp0));
		MultiAgentDPPlanningAgent a1 = new MultiAgentDPPlanningAgent(domain, vi, new PolicyFromJointPolicy(jp0));

		a0.joinWorld(w, at);
		a1.joinWorld(w, at);

		//run some games of the agents playing that policy
		GameAnalysis ga = null;
		for(int i = 0; i < 3; i++){
			ga = w.runGame();
		}

		//visualize results
		Visualizer v = GGVisualizer.getVisualizer(9, 9);
		new GameSequenceVisualizer(v, domain, Arrays.asList(ga));


	}

	public static void VICorrelatedTest(){

		GridGame gridGame = new GridGame();
		final OOSGDomain domain = gridGame.generateDomain();

		final HashableStateFactory hashingFactory = new SimpleHashableStateFactory();

		final State s = GridGame.getPrisonersDilemmaInitialState();

		JointReward rf = new GridGame.GGJointRewardFunction(domain, -1, 100, false);
		TerminalFunction tf = new GridGame.GGTerminalFunction(domain);

		SGAgentType at = GridGame.getStandardGridGameAgentType(domain);
		MAValueIteration vi = new MAValueIteration(domain, rf, tf, 0.99, hashingFactory, 0., new CorrelatedQ(CorrelatedEquilibriumSolver.CorrelatedEquilibriumObjective.UTILITARIAN), 0.00015, 50);

		World w = new World(domain, rf, tf, s);


		//for correlated Q, use a correlated equilibrium policy joint policy
		ECorrelatedQJointPolicy jp0 = new ECorrelatedQJointPolicy(CorrelatedEquilibriumSolver.CorrelatedEquilibriumObjective.UTILITARIAN, 0.);


		MultiAgentDPPlanningAgent a0 = new MultiAgentDPPlanningAgent(domain, vi, new PolicyFromJointPolicy(jp0, true));
		MultiAgentDPPlanningAgent a1 = new MultiAgentDPPlanningAgent(domain, vi, new PolicyFromJointPolicy(jp0, true));

		a0.joinWorld(w, at);
		a1.joinWorld(w, at);

		GameAnalysis ga = null;
		List<GameAnalysis> games = new ArrayList<GameAnalysis>();
		for(int i = 0; i < 10; i++){
			ga = w.runGame();
			games.add(ga);
		}

		Visualizer v = GGVisualizer.getVisualizer(9, 9);
		new GameSequenceVisualizer(v, domain, games);


	}

	public static void QLCoCoTest(){

		GridGame gridGame = new GridGame();
		final OOSGDomain domain = gridGame.generateDomain();

		final HashableStateFactory hashingFactory = new SimpleHashableStateFactory();

		final State s = GridGame.getPrisonersDilemmaInitialState();
		JointReward rf = new GridGame.GGJointRewardFunction(domain, -1, 100, false);
		TerminalFunction tf = new GridGame.GGTerminalFunction(domain);
		SGAgentType at = GridGame.getStandardGridGameAgentType(domain);

		World w = new World(domain, rf, tf, s);

		final double discount = 0.95;
		final double learningRate = 0.1;
		final double defaultQ = 100;

		MultiAgentQLearning a0 = new MultiAgentQLearning(domain, discount, learningRate, hashingFactory, defaultQ, new CoCoQ(), true);
		MultiAgentQLearning a1 = new MultiAgentQLearning(domain, discount, learningRate, hashingFactory, defaultQ, new CoCoQ(), true);

		a0.joinWorld(w, at);
		a1.joinWorld(w, at);


		//don't have the world print out debug info (comment out if you want to see it!)
		DPrint.toggleCode(w.getDebugId(), false);

		System.out.println("Starting training");
		int ngames = 1000;
		List<GameAnalysis> games = new ArrayList<GameAnalysis>();
		for(int i = 0; i < ngames; i++){
			GameAnalysis ga = w.runGame();
			games.add(ga);
			if(i % 10 == 0){
				System.out.println("Game: " + i + ": " + ga.maxTimeStep());
			}
		}

		System.out.println("Finished training");


		Visualizer v = GGVisualizer.getVisualizer(9, 9);
		new GameSequenceVisualizer(v, domain, games);

	}


	public static void saInterface(){

		GridGame gridGame = new GridGame();
		final OOSGDomain domain = gridGame.generateDomain();

		final HashableStateFactory hashingFactory = new SimpleHashableStateFactory();

		final State s = GridGame.getSimpleGameInitialState();
		JointReward rf = new GridGame.GGJointRewardFunction(domain, -1, 100, false);
		TerminalFunction tf = new GridGame.GGTerminalFunction(domain);
		SGAgentType at = GridGame.getStandardGridGameAgentType(domain);

		World w = new World(domain, rf, tf, s);

		//single agent Q-learning algorithms which will operate in our stochastic game
		//don't need to specify the domain, because the single agent interface will provide it
		QLearning ql1 = new QLearning(null, 0.99, new SimpleHashableStateFactory(), 0, 0.1);
		QLearning ql2 = new QLearning(null, 0.99, new SimpleHashableStateFactory(), 0, 0.1);

		//create a single-agent interface for each of our learning algorithm instances
		LearningAgentToSGAgentInterface a1 = new LearningAgentToSGAgentInterface(domain, ql1);
		LearningAgentToSGAgentInterface a2 = new LearningAgentToSGAgentInterface(domain, ql2);

		a1.joinWorld(w, at);
		a2.joinWorld(w, at);

		//don't have the world print out debug info (comment out if you want to see it!)
		DPrint.toggleCode(w.getDebugId(), false);

		System.out.println("Starting training");
		int ngames = 1000;
		List<GameAnalysis> gas = new ArrayList<GameAnalysis>(ngames);
		for(int i = 0; i < ngames; i++){
			GameAnalysis ga = w.runGame();
			gas.add(ga);
			if(i % 10 == 0){
				System.out.println("Game: " + i + ": " + ga.maxTimeStep());
			}
		}

		System.out.println("Finished training");


		Visualizer v = GGVisualizer.getVisualizer(9, 9);
		new GameSequenceVisualizer(v, domain, gas);


	}


	public static void main(String[] args) {

		//choose one

		VICoCoTest();
		//VICorrelatedTest();
		//QLCoCoTest();
		//saInterface();

	}

}
