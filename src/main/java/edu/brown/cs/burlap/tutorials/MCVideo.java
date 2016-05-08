package edu.brown.cs.burlap.tutorials;

import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.behavior.singleagent.vfa.common.NormalizedVariablesVectorGenerator;
import burlap.behavior.singleagent.vfa.fourier.FourierBasis;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.domain.singleagent.mountaincar.MCState;
import burlap.domain.singleagent.mountaincar.MountainCar;
import burlap.domain.singleagent.mountaincar.MountainCarVisualizer;
import burlap.mdp.auxiliary.StateGenerator;
import burlap.mdp.core.Domain;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.RewardFunction;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.EnvironmentServer;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.visualizer.Visualizer;


/**
 * @author James MacGlashan.
 */
public class MCVideo {

	public static void main(String[] args) {

		MountainCar mcGen = new MountainCar();
		Domain domain = mcGen.generateDomain();
		TerminalFunction tf = new MountainCar.ClassicMCTF();
		RewardFunction rf = new GoalBasedRF(tf, 100);

		StateGenerator rStateGen = new MCRandomStateGenerator(mcGen.physParams);
		SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
		SARSData dataset = collector.collectNInstances(rStateGen, rf, 5000, 20, tf, null);

		NormalizedVariablesVectorGenerator fvGen = new NormalizedVariablesVectorGenerator()
				.variableDomain("x", new VariableDomain(mcGen.physParams.xmin, mcGen.physParams.xmax))
				.variableDomain("v", new VariableDomain(mcGen.physParams.vmin, mcGen.physParams.vmax));
		FourierBasis fb = new FourierBasis(fvGen, 4);

		LSPI lspi = new LSPI(domain, 0.99, fb, dataset);
		Policy p = lspi.runPolicyIteration(30, 1e-6);

		Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
		VisualActionObserver vob = new VisualActionObserver(domain, v);
		vob.initGUI();

		SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf,
				new MCState(mcGen.physParams.valleyPos(), 0));
		EnvironmentServer envServ = new EnvironmentServer(env, vob);

		for(int i = 0; i < 100; i++){
			p.evaluateBehavior(envServ);
			envServ.resetEnvironment();
		}

		System.out.println("Finished");

	}

}
