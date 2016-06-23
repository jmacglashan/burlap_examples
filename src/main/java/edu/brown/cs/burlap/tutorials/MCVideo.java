package edu.brown.cs.burlap.tutorials;

import burlap.behavior.functionapproximation.dense.DenseCrossProductFeatures;
import burlap.behavior.functionapproximation.dense.NormalizedVariableFeatures;
import burlap.behavior.functionapproximation.dense.fourier.FourierBasis;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.domain.singleagent.mountaincar.MCState;
import burlap.domain.singleagent.mountaincar.MountainCar;
import burlap.domain.singleagent.mountaincar.MountainCarVisualizer;
import burlap.mdp.auxiliary.StateGenerator;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.environment.extensions.EnvironmentServer;
import burlap.visualizer.Visualizer;


/**
 * @author James MacGlashan.
 */
public class MCVideo {

	public static void main(String[] args) {

		MountainCar mcGen = new MountainCar();
		SADomain domain = mcGen.generateDomain();

		StateGenerator rStateGen = new MCRandomStateGenerator(mcGen.physParams);
		SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain);
		SARSData dataset = collector.collectNInstances(rStateGen, domain.getModel(), 5000, 20, null);

		NormalizedVariableFeatures features = new NormalizedVariableFeatures()
				.variableDomain("x", new VariableDomain(mcGen.physParams.xmin, mcGen.physParams.xmax))
				.variableDomain("v", new VariableDomain(mcGen.physParams.vmin, mcGen.physParams.vmax));
		FourierBasis fb = new FourierBasis(features, 4);

		LSPI lspi = new LSPI(domain, 0.99, new DenseCrossProductFeatures(fb, 3), dataset);
		Policy p = lspi.runPolicyIteration(30, 1e-6);

		Visualizer v = MountainCarVisualizer.getVisualizer(mcGen);
		VisualActionObserver vob = new VisualActionObserver(v);
		vob.initGUI();

		SimulatedEnvironment env = new SimulatedEnvironment(domain,
				new MCState(mcGen.physParams.valleyPos(), 0));
		EnvironmentServer envServ = new EnvironmentServer(env, vob);

		for(int i = 0; i < 100; i++){
			PolicyUtils.rollout(p, envServ);
			envServ.resetEnvironment();
		}

		System.out.println("Finished");

	}

}
