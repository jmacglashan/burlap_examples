package edu.brown.cs.burlap.tutorials;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.MDPSolver;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.valuefunction.ConstantValueFunction;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.QValue;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.statehashing.HashableState;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author James MacGlashan.
 */
public class QLTutorial extends MDPSolver implements LearningAgent, QProvider {

	Map<HashableState, List<QValue>> qValues;
	QFunction qinit;
	double learningRate;
	Policy learningPolicy;

	public QLTutorial(SADomain domain, double gamma, HashableStateFactory hashingFactory,
					  QFunction qinit, double learningRate, double epsilon){

		this.solverInit(domain, gamma, hashingFactory);
		this.qinit = qinit;
		this.learningRate = learningRate;
		this.qValues = new HashMap<HashableState, List<QValue>>();
		this.learningPolicy = new EpsilonGreedy(this, epsilon);

	}

	@Override
	public Episode runLearningEpisode(Environment env) {
		return this.runLearningEpisode(env, -1);
	}

	@Override
	public Episode runLearningEpisode(Environment env, int maxSteps) {
		//initialize our episode object with the initial state of the environment
		Episode e = new Episode(env.currentObservation());

		//behave until a terminal state or max steps is reached
		State curState = env.currentObservation();
		int steps = 0;
		while(!env.isInTerminalState() && (steps < maxSteps || maxSteps == -1)){

			//select an action
			Action a = this.learningPolicy.action(curState);

			//take the action and observe outcome
			EnvironmentOutcome eo = env.executeAction(a);

			//record result
			e.transition(eo);

			//get the max Q value of the resulting state if it's not terminal, 0 otherwise
			double maxQ = eo.terminated ? 0. : this.value(eo.op);

			//update the old Q-value
			QValue oldQ = this.storedQ(curState, a);
			oldQ.q = oldQ.q + this.learningRate * (eo.r + this.gamma * maxQ - oldQ.q);


			//update state pointer to next environment state observed
			curState = eo.op;
			steps++;

		}

		return e;
	}

	@Override
	public void resetSolver() {
		this.qValues.clear();
	}

	@Override
	public List<QValue> qValues(State s) {
		//first get hashed state
		HashableState sh = this.hashingFactory.hashState(s);

		//check if we already have stored values
		List<QValue> qs = this.qValues.get(sh);

		//create and add initialized Q-values if we don't have them stored for this state
		if(qs == null){
			List<Action> actions = this.applicableActions(s);
			qs = new ArrayList<QValue>(actions.size());
			//create a Q-value for each action
			for(Action a : actions){
				//add q with initialized value
				qs.add(new QValue(s, a, this.qinit.qValue(s, a)));
			}
			//store this for later
			this.qValues.put(sh, qs);
		}

		return qs;
	}

	@Override
	public double qValue(State s, Action a) {
		return storedQ(s, a).q;
	}


	protected QValue storedQ(State s, Action a){
		//first get all Q-values
		List<QValue> qs = this.qValues(s);

		//iterate through stored Q-values to find a match for the input action
		for(QValue q : qs){
			if(q.a.equals(a)){
				return q;
			}
		}

		throw new RuntimeException("Could not find matching Q-value.");
	}

	@Override
	public double value(State s) {
		return QProvider.Helper.maxQ(this, s);
	}


	public static void main(String[] args) {

		GridWorldDomain gwd = new GridWorldDomain(11, 11);
		gwd.setMapToFourRooms();
		gwd.setProbSucceedTransitionDynamics(0.8);
		gwd.setTf(new GridWorldTerminalFunction(10, 10));

		SADomain domain = gwd.generateDomain();

		//get initial state with agent in 0,0
		State s = new GridWorldState(new GridAgent(0, 0));

		//create environment
		SimulatedEnvironment env = new SimulatedEnvironment(domain, s);

		//create Q-learning
		QLTutorial agent = new QLTutorial(domain, 0.99, new SimpleHashableStateFactory(),
				new ConstantValueFunction(), 0.1, 0.1);

		//run Q-learning and store results in a list
		List<Episode> episodes = new ArrayList<Episode>(1000);
		for(int i = 0; i < 1000; i++){
			episodes.add(agent.runLearningEpisode(env));
			env.resetEnvironment();
		}

		Visualizer v = GridWorldVisualizer.getVisualizer(gwd.getMap());
		new EpisodeSequenceVisualizer(v, domain, episodes);

	}

}
