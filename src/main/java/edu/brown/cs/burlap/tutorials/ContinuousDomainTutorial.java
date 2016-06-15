package edu.brown.cs.burlap.tutorials;

import burlap.behavior.functionapproximation.DifferentiableStateActionValue;
import burlap.behavior.functionapproximation.dense.ConcatenatedObjectFeatures;
import burlap.behavior.functionapproximation.dense.DenseCrossProductFeatures;
import burlap.behavior.functionapproximation.dense.NormalizedVariableFeatures;
import burlap.behavior.functionapproximation.dense.NumericVariableFeatures;
import burlap.behavior.functionapproximation.dense.fourier.FourierBasis;
import burlap.behavior.functionapproximation.dense.rbf.DistanceMetric;
import burlap.behavior.functionapproximation.dense.rbf.RBFFeatures;
import burlap.behavior.functionapproximation.dense.rbf.functions.GaussianRBF;
import burlap.behavior.functionapproximation.dense.rbf.metrics.EuclideanDistance;
import burlap.behavior.functionapproximation.sparse.tilecoding.TileCodingFeatures;
import burlap.behavior.functionapproximation.sparse.tilecoding.TilingArrangement;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.gridset.FlatStateGridder;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
import burlap.behavior.singleagent.planning.stochastic.sparsesampling.SparseSampling;
import burlap.domain.singleagent.cartpole.CartPoleVisualizer;
import burlap.domain.singleagent.cartpole.InvertedPendulum;
import burlap.domain.singleagent.cartpole.states.InvertedPendulumState;
import burlap.domain.singleagent.lunarlander.LLVisualizer;
import burlap.domain.singleagent.lunarlander.LunarLanderDomain;
import burlap.domain.singleagent.lunarlander.state.LLAgent;
import burlap.domain.singleagent.lunarlander.state.LLBlock;
import burlap.domain.singleagent.lunarlander.state.LLState;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.domain.singleagent.mountaincar.MCState;
import burlap.domain.singleagent.mountaincar.MountainCar;
import burlap.domain.singleagent.mountaincar.MountainCarVisualizer;
import burlap.mdp.auxiliary.StateGenerator;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

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
		SADomain domain = mcGen.generateDomain();

		StateGenerator rStateGen = new MCRandomStateGenerator(mcGen.physParams);
		SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
		SARSData dataset = collector.collectNInstances(rStateGen, domain.getModel(), 5000, 20, null);

		NormalizedVariableFeatures inputFeatures = new NormalizedVariableFeatures()
				.variableDomain("x", new VariableDomain(mcGen.physParams.xmin, mcGen.physParams.xmax))
				.variableDomain("v", new VariableDomain(mcGen.physParams.vmin, mcGen.physParams.vmax));

		FourierBasis fb = new FourierBasis(inputFeatures, 4);

		LSPI lspi = new LSPI(domain, 0.99, new DenseCrossProductFeatures(fb, 3), dataset);
		Policy p = lspi.runPolicyIteration(30, 1e-6);

		Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
		VisualActionObserver vob = new VisualActionObserver(v);
		vob.initGUI();

		SimulatedEnvironment env = new SimulatedEnvironment(domain, new MCState(mcGen.physParams.valleyPos(), 0.));
		env.addObservers(vob);

		for(int i = 0; i < 5; i++){
			PolicyUtils.rollout(p, env);
			env.resetEnvironment();
		}

		System.out.println("Finished");


	}

	public static void MCLSPIRBF(){

		MountainCar mcGen = new MountainCar();
		SADomain domain = mcGen.generateDomain();
		MCState s = new MCState(mcGen.physParams.valleyPos(), 0.);

		NormalizedVariableFeatures inputFeatures = new NormalizedVariableFeatures()
				.variableDomain("x", new VariableDomain(mcGen.physParams.xmin, mcGen.physParams.xmax))
				.variableDomain("v", new VariableDomain(mcGen.physParams.vmin, mcGen.physParams.vmax));

		StateGenerator rStateGen = new MCRandomStateGenerator(mcGen.physParams);
		SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
		SARSData dataset = collector.collectNInstances(rStateGen, domain.getModel(), 5000, 20, null);

		RBFFeatures rbf = new RBFFeatures(inputFeatures, true);
		FlatStateGridder gridder = new FlatStateGridder()
				.gridDimension("x", mcGen.physParams.xmin, mcGen.physParams.xmax, 5)
				.gridDimension("v", mcGen.physParams.vmin, mcGen.physParams.vmax, 5);

		List<State> griddedStates = gridder.gridState(s);
		DistanceMetric metric = new EuclideanDistance();
		for(State g : griddedStates){
			rbf.addRBF(new GaussianRBF(inputFeatures.features(g), metric, 0.2));
		}

		LSPI lspi = new LSPI(domain, 0.99, new DenseCrossProductFeatures(rbf, 3), dataset);
		Policy p = lspi.runPolicyIteration(30, 1e-6);

		Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
		VisualActionObserver vob = new VisualActionObserver(v);
		vob.initGUI();


		SimulatedEnvironment env = new SimulatedEnvironment(domain, s);
		env.addObservers(vob);

		for(int i = 0; i < 5; i++){
			PolicyUtils.rollout(p, env);
			env.resetEnvironment();
		}

		System.out.println("Finished");


	}


	public static void IPSS(){

		InvertedPendulum ip = new InvertedPendulum();
		ip.physParams.actionNoise = 0.;
		RewardFunction rf = new InvertedPendulum.InvertedPendulumRewardFunction(Math.PI/8.);
		TerminalFunction tf = new InvertedPendulum.InvertedPendulumTerminalFunction(Math.PI/8.);
		ip.setRf(rf);
		ip.setTf(tf);
		SADomain domain = ip.generateDomain();

		State initialState = new InvertedPendulumState();

		SparseSampling ss = new SparseSampling(domain, 1, new SimpleHashableStateFactory(), 10, 1);
		ss.setForgetPreviousPlanResults(true);
		ss.toggleDebugPrinting(false);
		Policy p = new GreedyQPolicy(ss);

		Episode e = PolicyUtils.rollout(p, initialState, domain.getModel(), 500);
		System.out.println("Num steps: " + e.maxTimeStep());
		Visualizer v = CartPoleVisualizer.getCartPoleVisualizer();
		new EpisodeSequenceVisualizer(v, domain, Arrays.asList(e));

	}

	public static void LLSARSA(){

		LunarLanderDomain lld = new LunarLanderDomain();
		OOSADomain domain = lld.generateDomain();

		LLState s = new LLState(new LLAgent(5, 0, 0), new LLBlock.LLPad(75, 95, 0, 10, "pad"));

		ConcatenatedObjectFeatures inputFeatures = new ConcatenatedObjectFeatures()
				.addObjectVectorizion(LunarLanderDomain.CLASS_AGENT, new NumericVariableFeatures());

		int nTilings = 5;
		double resolution = 10.;

		double xWidth = (lld.getXmax() - lld.getXmin()) / resolution;
		double yWidth = (lld.getYmax() - lld.getYmin()) / resolution;
		double velocityWidth = 2 * lld.getVmax() / resolution;
		double angleWidth = 2 * lld.getAngmax() / resolution;



		TileCodingFeatures tilecoding = new TileCodingFeatures(inputFeatures);
		tilecoding.addTilingsForAllDimensionsWithWidths(
				new double []{xWidth, yWidth, velocityWidth, velocityWidth, angleWidth},
				nTilings,
				TilingArrangement.RANDOM_JITTER);




		double defaultQ = 0.5;
		DifferentiableStateActionValue vfa = tilecoding.generateVFA(defaultQ/nTilings);
		GradientDescentSarsaLam agent = new GradientDescentSarsaLam(domain, 0.99, vfa, 0.02, 0.5);

		SimulatedEnvironment env = new SimulatedEnvironment(domain, s);
		List<Episode> episodes = new ArrayList<Episode>();
		for(int i = 0; i < 5000; i++){
			Episode ea = agent.runLearningEpisode(env);
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
