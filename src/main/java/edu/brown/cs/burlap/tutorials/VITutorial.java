package edu.brown.cs.burlap.tutorials;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.MDPSolver;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.QValue;
import burlap.behavior.valuefunction.ValueFunctionInitialization;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.AbstractGroundedAction;
import burlap.mdp.core.Domain;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.TransitionProbability;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.GroundedAction;
import burlap.mdp.singleagent.RewardFunction;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.common.UniformCostRF;
import burlap.mdp.statehashing.HashableState;
import burlap.mdp.statehashing.HashableStateFactory;
import burlap.mdp.statehashing.SimpleHashableStateFactory;
import burlap.mdp.visualizer.Visualizer;

import java.util.*;

/**
 * @author James MacGlashan.
 */
public class VITutorial extends MDPSolver implements Planner, QFunction {

	protected Map<HashableState, Double> valueFunction;
	protected ValueFunctionInitialization vinit;
	protected int numIterations;


	public VITutorial(Domain domain, RewardFunction rf, TerminalFunction tf, double gamma,
					  HashableStateFactory hashingFactory, ValueFunctionInitialization vinit, int numIterations){
		this.solverInit(domain, rf, tf, gamma, hashingFactory);
		this.vinit = vinit;
		this.numIterations = numIterations;
		this.valueFunction = new HashMap<HashableState, Double>();
	}

	public double value(State s) {
		Double d = this.valueFunction.get(hashingFactory.hashState(s));
		if(d == null){
			return vinit.value(s);
		}
		return d;
	}

	public List<QValue> getQs(State s) {
		List<GroundedAction> applicableActions = this.getAllGroundedActions(s);
		List<QValue> qs = new ArrayList<QValue>(applicableActions.size());
		for(GroundedAction ga : applicableActions){
			qs.add(this.getQ(s, ga));
		}
		return qs;
	}

	public QValue getQ(State s, AbstractGroundedAction a) {

		//type cast to the type we're using
		GroundedAction ga = (GroundedAction)a;

		//what are the possible outcomes?
		List<TransitionProbability> tps = ga.getTransitions(s);

		//aggregate over each possible outcome
		double q = 0.;
		for(TransitionProbability tp : tps){
			//what is reward for this transition?
			double r = this.rf.reward(s, ga, tp.s);

			//what is the value for the next state?
			double vp = this.valueFunction.get(this.hashingFactory.hashState(tp.s));

			//add contribution weighted by transition probabiltiy and
			//discounting the next state
			q += tp.p * (r + this.gamma * vp);
		}

		//create Q-value wrapper
		QValue qValue = new QValue(s, ga, q);

		return qValue;
	}

	protected double bellmanEquation(State s){

		if(this.tf.isTerminal(s)){
			return 0.;
		}

		List<QValue> qs = this.getQs(s);
		double maxQ = Double.NEGATIVE_INFINITY;
		for(QValue q : qs){
			maxQ = Math.max(maxQ, q.q);
		}
		return maxQ;
	}

	public GreedyQPolicy planFromState(State initialState) {

		HashableState hashedInitialState = this.hashingFactory.hashState(initialState);
		if(this.valueFunction.containsKey(hashedInitialState)){
			return new GreedyQPolicy(this); //already performed planning here!
		}

		//if the state is new, then find all reachable states from it first
		this.performReachabilityFrom(initialState);

		//now perform multiple iterations over the whole state space
		for(int i = 0; i < this.numIterations; i++){
			//iterate over each state
			for(HashableState sh : this.valueFunction.keySet()){
				//update its value using the bellman equation
				this.valueFunction.put(sh, this.bellmanEquation(sh.s));
			}
		}

		return new GreedyQPolicy(this);

	}

	@Override
	public void resetSolver() {

	}

	public void performReachabilityFrom(State seedState){

		Set<HashableState> hashedStates = StateReachability.getReachableHashedStates(seedState, (SADomain) this.domain, this.hashingFactory);

		//initialize the value function for all states
		for(HashableState hs : hashedStates){
			if(!this.valueFunction.containsKey(hs)){
				this.valueFunction.put(hs, this.vinit.value(hs.s));
			}
		}

	}


	public static void main(String [] args){

		GridWorldDomain gwd = new GridWorldDomain(11, 11);
		gwd.setMapToFourRooms();

		//only go in intended directon 80% of the time
		gwd.setProbSucceedTransitionDynamics(0.8);

		Domain domain = gwd.generateDomain();

		//get initial state with agent in 0,0
		State s = new GridWorldState(new GridAgent(0, 0), new GridLocation(10, 10, "loc0"));

		//all transitions return -1
		RewardFunction rf = new UniformCostRF();

		//terminate in top right corner
		TerminalFunction tf = new GridWorldTerminalFunction(10, 10);

		//setup vi with 0.99 discount factor, a value
		//function initialization that initializes all states to value 0, and which will
		//run for 30 iterations over the state space
		VITutorial vi = new VITutorial(domain, rf, tf, 0.99, new SimpleHashableStateFactory(),
				new ValueFunctionInitialization.ConstantValueFunctionInitialization(0.0), 30);

		//run planning from our initial state
		Policy p = vi.planFromState(s);

		//evaluate the policy with one roll out visualize the trajectory
		EpisodeAnalysis ea = p.evaluateBehavior(s, rf, tf);

		Visualizer v = GridWorldVisualizer.getVisualizer(gwd.getMap());
		new EpisodeSequenceVisualizer(v, domain, Arrays.asList(ea));

	}

}
