package edu.brown.cs.burlap.tutorials;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.gridset.FlatStateGridder;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
import burlap.behavior.singleagent.planning.stochastic.sparsesampling.SparseSampling;
import burlap.behavior.singleagent.vfa.DifferentiableStateActionValue;
import burlap.behavior.singleagent.vfa.common.ConcatenatedObjectFeatureVectorGenerator;
import burlap.behavior.singleagent.vfa.common.NormalizedVariablesVectorGenerator;
import burlap.behavior.singleagent.vfa.common.VariablesVectorGenerator;
import burlap.behavior.singleagent.vfa.fourier.FourierBasis;
import burlap.behavior.singleagent.vfa.rbf.DistanceMetric;
import burlap.behavior.singleagent.vfa.rbf.RBFFeatures;
import burlap.behavior.singleagent.vfa.rbf.functions.GaussianRBF;
import burlap.behavior.singleagent.vfa.rbf.metrics.EuclideanDistance;
import burlap.behavior.singleagent.vfa.tilecoding.TileCodingFeatures;
import burlap.behavior.singleagent.vfa.tilecoding.TilingArrangement;
import burlap.domain.singleagent.cartpole.CartPoleVisualizer;
import burlap.domain.singleagent.cartpole.InvertedPendulum;
import burlap.domain.singleagent.cartpole.states.InvertedPendulumState;
import burlap.domain.singleagent.lunarlander.LLVisualizer;
import burlap.domain.singleagent.lunarlander.LunarLanderDomain;
import burlap.domain.singleagent.lunarlander.LunarLanderRF;
import burlap.domain.singleagent.lunarlander.LunarLanderTF;
import burlap.domain.singleagent.lunarlander.state.LLAgent;
import burlap.domain.singleagent.lunarlander.state.LLBlock;
import burlap.domain.singleagent.lunarlander.state.LLState;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.domain.singleagent.mountaincar.MCState;
import burlap.domain.singleagent.mountaincar.MountainCar;
import burlap.domain.singleagent.mountaincar.MountainCarVisualizer;
import burlap.mdp.auxiliary.StateGenerator;
import burlap.mdp.core.Domain;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.RewardFunction;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.mdp.statehashing.SimpleHashableStateFactory;
import burlap.mdp.visualizer.Visualizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author James MacGlashan.
 */
public class ContinuousDomainTutorial {

	private ContinuousDomainTutorial() {
		// do nothing
	}

	public static void MCLSPIFB(){

		MountainCar mcGen = new MountainCar();
		Domain domain = mcGen.generateDomain();
		TerminalFunction tf = new MountainCar.ClassicMCTF();
		RewardFunction rf = new GoalBasedRF(tf, 100);

		StateGenerator rStateGen = new MCRandomStateGenerator(mcGen.physParams);
		SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
		SARSData dataset = collector.collectNInstances(rStateGen, rf, 5000, 20, tf, null);

		NormalizedVariablesVectorGenerator featureVectorGenerator = new NormalizedVariablesVectorGenerator()
				.variableDomain("x", new VariableDomain(mcGen.physParams.xmin, mcGen.physParams.xmax))
				.variableDomain("v", new VariableDomain(mcGen.physParams.vmin, mcGen.physParams.vmax));

		FourierBasis fb = new FourierBasis(featureVectorGenerator, 4);

		LSPI lspi = new LSPI(domain, 0.99, fb, dataset);
		Policy p = lspi.runPolicyIteration(30, 1e-6);

		Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
		VisualActionObserver vob = new VisualActionObserver(domain, v);
		vob.initGUI();

		SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, new MCState(mcGen.physParams.valleyPos(), 0.));
		env.addObservers(vob);

		for(int i = 0; i < 5; i++){
			p.evaluateBehavior(env);
			env.resetEnvironment();
		}

		System.out.println("Finished");


	}

	public static void MCLSPIRBF(){

		MountainCar mcGen = new MountainCar();
		Domain domain = mcGen.generateDomain();
		TerminalFunction tf = new MountainCar.ClassicMCTF();
		RewardFunction rf = new GoalBasedRF(tf, 100);
		MCState s = new MCState(mcGen.physParams.valleyPos(), 0.);

		NormalizedVariablesVectorGenerator normVariables = new NormalizedVariablesVectorGenerator()
				.variableDomain("x", new VariableDomain(mcGen.physParams.xmin, mcGen.physParams.xmax))
				.variableDomain("v", new VariableDomain(mcGen.physParams.vmin, mcGen.physParams.vmax));

		StateGenerator rStateGen = new MCRandomStateGenerator(mcGen.physParams);
		SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
		SARSData dataset = collector.collectNInstances(rStateGen, rf, 5000, 20, tf, null);

		RBFFeatures rbf = new RBFFeatures(normVariables, true);
		FlatStateGridder gridder = new FlatStateGridder()
				.gridDimension("x", mcGen.physParams.xmin, mcGen.physParams.xmax, 5)
				.gridDimension("v", mcGen.physParams.vmin, mcGen.physParams.vmax, 5);

		List<State> griddedStates = gridder.gridState(s);
		DistanceMetric metric = new EuclideanDistance();
		for(State g : griddedStates){
			rbf.addRBF(new GaussianRBF(normVariables.generateFeatureVectorFrom(g), metric, 0.2));
		}

		LSPI lspi = new LSPI(domain, 0.99, rbf, dataset);
		Policy p = lspi.runPolicyIteration(30, 1e-6);

		Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
		VisualActionObserver vob = new VisualActionObserver(domain, v);
		vob.initGUI();


		SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, s);
		env.addObservers(vob);

		for(int i = 0; i < 5; i++){
			p.evaluateBehavior(env);
			env.resetEnvironment();
		}

		System.out.println("Finished");


	}


	public static void IPSS(){

		InvertedPendulum ip = new InvertedPendulum();
		ip.physParams.actionNoise = 0.;
		Domain domain = ip.generateDomain();
		RewardFunction rf = new InvertedPendulum.InvertedPendulumRewardFunction(Math.PI/8.);
		TerminalFunction tf = new InvertedPendulum.InvertedPendulumTerminalFunction(Math.PI/8.);
		State initialState = new InvertedPendulumState();

		SparseSampling ss = new SparseSampling(domain, rf, tf, 1, new SimpleHashableStateFactory(), 10 ,1);
		ss.setForgetPreviousPlanResults(true);
		ss.toggleDebugPrinting(false);
		Policy p = new GreedyQPolicy(ss);

		EpisodeAnalysis ea = p.evaluateBehavior(initialState, rf, tf, 500);
		System.out.println("Num steps: " + ea.maxTimeStep());
		Visualizer v = CartPoleVisualizer.getCartPoleVisualizer();
		new EpisodeSequenceVisualizer(v, domain, Arrays.asList(ea));

	}

	public static void LLSARSA(){

		LunarLanderDomain lld = new LunarLanderDomain();
		OOSADomain domain = (OOSADomain)lld.generateDomain();
		RewardFunction rf = new LunarLanderRF(domain);
		TerminalFunction tf = new LunarLanderTF(domain);

		LLState s = new LLState(new LLAgent(0, 5, 0), new LLBlock.LLPad(75, 95, 0, 10, "pad"));

		ConcatenatedObjectFeatureVectorGenerator stateVars = new ConcatenatedObjectFeatureVectorGenerator()
				.addObjectVectorizion(LunarLanderDomain.CLASS_AGENT, new VariablesVectorGenerator());

		int nTilings = 5;
		double resolution = 10.;

		double xWidth = (lld.getXmax() - lld.getXmin()) / resolution;
		double yWidth = (lld.getYmax() - lld.getYmin()) / resolution;
		double velocityWidth = 2 * lld.getVmax() / resolution;
		double angleWidth = 2 * lld.getAngmax() / resolution;



		TileCodingFeatures cmac = new TileCodingFeatures(stateVars);
		cmac.addTilingsForAllDimensionsWithWidths(
				new double []{xWidth, yWidth, velocityWidth, velocityWidth, angleWidth},
				nTilings,
				TilingArrangement.RANDOMJITTER);




		double defaultQ = 0.5;
		DifferentiableStateActionValue vfa = cmac.generateVFA(defaultQ/nTilings);
		GradientDescentSarsaLam agent = new GradientDescentSarsaLam(domain, 0.99, vfa, 0.02, 0.5);

		SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, s);
		List<EpisodeAnalysis> episodes = new ArrayList<EpisodeAnalysis>();
		for(int i = 0; i < 5000; i++){
			EpisodeAnalysis ea = agent.runLearningEpisode(env);
			episodes.add(ea);
			System.out.println(i + ": " + ea.maxTimeStep());
			env.resetEnvironment();
		}

		Visualizer v = LLVisualizer.getVisualizer(lld.getPhysParams());
		new EpisodeSequenceVisualizer(v, domain, episodes);

	}


	public static void main(String[] args) {
		//MCLSPIFB();
		//MCLSPIRBF();
		//IPSS();
		LLSARSA();
	}

}
