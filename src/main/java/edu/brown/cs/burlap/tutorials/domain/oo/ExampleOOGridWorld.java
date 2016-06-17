package edu.brown.cs.burlap.tutorials.domain.oo;

import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.auxiliary.common.SinglePFTF;
import burlap.mdp.core.StateTransitionProb;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.UniversalActionType;
import burlap.mdp.core.oo.OODomain;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.oo.state.ObjectInstance;
import burlap.mdp.core.oo.state.generic.GenericOOState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.SingleGoalPFRF;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.model.statemodel.FullStateModel;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.visualizer.*;

import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author James MacGlashan.
 */
public class ExampleOOGridWorld implements DomainGenerator{

	public static final String VAR_X = "x";
	public static final String VAR_Y = "y";
	public static final String VAR_TYPE = "type";

	public static final String CLASS_AGENT = "agent";
	public static final String CLASS_LOCATION = "location";

	public static final String ACTION_NORTH = "north";
	public static final String ACTION_SOUTH = "south";
	public static final String ACTION_EAST = "east";
	public static final String ACTION_WEST = "west";

	public static final String PF_AT = "at";


	//ordered so first dimension is x
	protected int [][] map = new int[][]{
			{0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,1,0,0,0,0,0},
			{1,0,1,1,1,1,1,1,0,1,1},
			{0,0,0,0,1,0,0,0,0,0,0},
			{0,0,0,0,1,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,1,0,0,0,0,0,0},
			{0,0,0,0,1,0,0,0,0,0,0},
	};

	public List<PropositionalFunction> generatePfs(){
		return Arrays.<PropositionalFunction>asList(new AtLocation());
	}

	@Override
	public OOSADomain generateDomain() {

		OOSADomain domain = new OOSADomain();

		domain.addStateClass(CLASS_AGENT, ExGridAgent.class)
				.addStateClass(CLASS_LOCATION, EXGridLocation.class);

		domain.addActionTypes(
				new UniversalActionType(ACTION_NORTH),
				new UniversalActionType(ACTION_SOUTH),
				new UniversalActionType(ACTION_EAST),
				new UniversalActionType(ACTION_WEST));


		OODomain.Helper.addPfsToDomain(domain, this.generatePfs());

		OOGridWorldStateModel smodel = new OOGridWorldStateModel();
		RewardFunction rf = new SingleGoalPFRF(domain.propFunction(PF_AT), 100, -1);
		TerminalFunction tf = new SinglePFTF(domain.propFunction(PF_AT));

		domain.setModel(new FactoredModel(smodel, rf, tf));


		return domain;
	}


	protected class OOGridWorldStateModel implements FullStateModel {


		protected double [][] transitionProbs;

		public OOGridWorldStateModel() {
			this.transitionProbs = new double[4][4];
			for(int i = 0; i < 4; i++){
				for(int j = 0; j < 4; j++){
					double p = i != j ? 0.2/3 : 0.8;
					transitionProbs[i][j] = p;
				}
			}
		}

		public List<StateTransitionProb> stateTransitions(State s, Action a) {

			//get agent current position
			GenericOOState gs = (GenericOOState)s;
			ExGridAgent agent = (ExGridAgent)gs.object(CLASS_AGENT);

			int curX = agent.x;
			int curY = agent.y;

			int adir = actionDir(a);

			List<StateTransitionProb> tps = new ArrayList<StateTransitionProb>(4);
			StateTransitionProb noChange = null;
			for(int i = 0; i < 4; i++){

				int [] newPos = this.moveResult(curX, curY, i);
				if(newPos[0] != curX || newPos[1] != curY){
					//new possible outcome
					GenericOOState ns = gs.copy();
					ExGridAgent nagent = (ExGridAgent)ns.touch(CLASS_AGENT);
					nagent.x = newPos[0];
					nagent.y = newPos[1];

					//create transition probability object and add to our list of outcomes
					tps.add(new StateTransitionProb(ns, this.transitionProbs[adir][i]));
				}
				else{
					//this direction didn't lead anywhere new
					//if there are existing possible directions
					//that wouldn't lead anywhere, aggregate with them
					if(noChange != null){
						noChange.p += this.transitionProbs[adir][i];
					}
					else{
						//otherwise create this new state and transition
						noChange = new StateTransitionProb(s.copy(), this.transitionProbs[adir][i]);
						tps.add(noChange);
					}
				}

			}


			return tps;
		}

		public State sample(State s, Action a) {

			s = s.copy();
			GenericOOState gs = (GenericOOState)s;
			ExGridAgent agent = (ExGridAgent)gs.touch(CLASS_AGENT);
			int curX = agent.x;
			int curY = agent.y;

			int adir = actionDir(a);

			//sample direction with random roll
			double r = Math.random();
			double sumProb = 0.;
			int dir = 0;
			for(int i = 0; i < 4; i++){
				sumProb += this.transitionProbs[adir][i];
				if(r < sumProb){
					dir = i;
					break; //found direction
				}
			}

			//get resulting position
			int [] newPos = this.moveResult(curX, curY, dir);

			//set the new position
			agent.x = newPos[0];
			agent.y = newPos[1];

			//return the state we just modified
			return gs;
		}

		protected int actionDir(Action a){
			int adir = -1;
			if(a.actionName().equals(ACTION_NORTH)){
				adir = 0;
			}
			else if(a.actionName().equals(ACTION_SOUTH)){
				adir = 1;
			}
			else if(a.actionName().equals(ACTION_EAST)){
				adir = 2;
			}
			else if(a.actionName().equals(ACTION_WEST)){
				adir = 3;
			}
			return adir;
		}


		protected int [] moveResult(int curX, int curY, int direction){

			//first get change in x and y from direction using 0: north; 1: south; 2:east; 3: west
			int xdelta = 0;
			int ydelta = 0;
			if(direction == 0){
				ydelta = 1;
			}
			else if(direction == 1){
				ydelta = -1;
			}
			else if(direction == 2){
				xdelta = 1;
			}
			else{
				xdelta = -1;
			}

			int nx = curX + xdelta;
			int ny = curY + ydelta;

			int width = ExampleOOGridWorld.this.map.length;
			int height = ExampleOOGridWorld.this.map[0].length;

			//make sure new position is valid (not a wall or off bounds)
			if(nx < 0 || nx >= width || ny < 0 || ny >= height ||
					ExampleOOGridWorld.this.map[nx][ny] == 1){
				nx = curX;
				ny = curY;
			}


			return new int[]{nx,ny};

		}
	}


	public Visualizer getVisualizer(){
		return new Visualizer(this.getStateRenderLayer());
	}

	public StateRenderLayer getStateRenderLayer(){
		StateRenderLayer rl = new StateRenderLayer();
		rl.addStatePainter(new ExampleOOGridWorld.WallPainter());
		OOStatePainter ooStatePainter = new OOStatePainter();
		ooStatePainter.addObjectClassPainter(CLASS_LOCATION, new LocationPainter());
		ooStatePainter.addObjectClassPainter(CLASS_AGENT, new AgentPainter());
		rl.addStatePainter(ooStatePainter);


		return rl;
	}


	protected class AtLocation extends PropositionalFunction {

		public AtLocation(){
			super(PF_AT, new String []{CLASS_AGENT, CLASS_LOCATION});
		}

		@Override
		public boolean isTrue(OOState s, String... params) {
			ObjectInstance agent = s.object(params[0]);
			ObjectInstance location = s.object(params[1]);

			int ax = (Integer)agent.get(VAR_X);
			int ay = (Integer)agent.get(VAR_Y);

			int lx = (Integer)location.get(VAR_X);
			int ly = (Integer)location.get(VAR_Y);

			return ax == lx && ay == ly;

		}

	}



	public class WallPainter implements StatePainter {

		public void paint(Graphics2D g2, State s, float cWidth, float cHeight) {

			//walls will be filled in black
			g2.setColor(Color.BLACK);

			//set up floats for the width and height of our domain
			float fWidth = ExampleOOGridWorld.this.map.length;
			float fHeight = ExampleOOGridWorld.this.map[0].length;

			//determine the width of a single cell
			//on our canvas such that the whole map can be painted
			float width = cWidth / fWidth;
			float height = cHeight / fHeight;

			//pass through each cell of our map and if it's a wall, paint a black rectangle on our
			//cavas of dimension widthxheight
			for(int i = 0; i < ExampleOOGridWorld.this.map.length; i++){
				for(int j = 0; j < ExampleOOGridWorld.this.map[0].length; j++){

					//is there a wall here?
					if(ExampleOOGridWorld.this.map[i][j] == 1){

						//left coordinate of cell on our canvas
						float rx = i*width;

						//top coordinate of cell on our canvas
						//coordinate system adjustment because the java canvas
						//origin is in the top left instead of the bottom right
						float ry = cHeight - height - j*height;

						//paint the rectangle
						g2.fill(new Rectangle2D.Float(rx, ry, width, height));

					}


				}
			}

		}


	}


	public class AgentPainter implements ObjectPainter {

		@Override
		public void paintObject(Graphics2D g2, OOState s, ObjectInstance ob,
								float cWidth, float cHeight) {

			//agent will be filled in gray
			g2.setColor(Color.GRAY);

			//set up floats for the width and height of our domain
			float fWidth = ExampleOOGridWorld.this.map.length;
			float fHeight = ExampleOOGridWorld.this.map[0].length;

			//determine the width of a single cell on our canvas
			//such that the whole map can be painted
			float width = cWidth / fWidth;
			float height = cHeight / fHeight;

			int ax = (Integer)ob.get(VAR_X);
			int ay = (Integer)ob.get(VAR_Y);

			//left coordinate of cell on our canvas
			float rx = ax*width;

			//top coordinate of cell on our canvas
			//coordinate system adjustment because the java canvas
			//origin is in the top left instead of the bottom right
			float ry = cHeight - height - ay*height;

			//paint the rectangle
			g2.fill(new Ellipse2D.Float(rx, ry, width, height));


		}



	}

	public class LocationPainter implements ObjectPainter {

		@Override
		public void paintObject(Graphics2D g2, OOState s, ObjectInstance ob,
								float cWidth, float cHeight) {

			//agent will be filled in blue
			g2.setColor(Color.BLUE);

			//set up floats for the width and height of our domain
			float fWidth = ExampleOOGridWorld.this.map.length;
			float fHeight = ExampleOOGridWorld.this.map[0].length;

			//determine the width of a single cell on our canvas
			//such that the whole map can be painted
			float width = cWidth / fWidth;
			float height = cHeight / fHeight;

			int ax = (Integer)ob.get(VAR_X);
			int ay = (Integer)ob.get(VAR_Y);

			//left coordinate of cell on our canvas
			float rx = ax*width;

			//top coordinate of cell on our canvas
			//coordinate system adjustment because the java canvas
			//origin is in the top left instead of the bottom right
			float ry = cHeight - height - ay*height;

			//paint the rectangle
			g2.fill(new Rectangle2D.Float(rx, ry, width, height));


		}



	}


	public static void main(String [] args){

		ExampleOOGridWorld gen = new ExampleOOGridWorld();
		OOSADomain domain = gen.generateDomain();
		State initialState = new GenericOOState(new ExGridAgent(0, 0), new EXGridLocation(10, 10, "loc0"));
		SimulatedEnvironment env = new SimulatedEnvironment(domain, initialState);

		Visualizer v = gen.getVisualizer();
		VisualExplorer exp = new VisualExplorer(domain, env, v);

		exp.addKeyAction("w", ACTION_NORTH, "");
		exp.addKeyAction("s", ACTION_SOUTH, "");
		exp.addKeyAction("d", ACTION_EAST, "");
		exp.addKeyAction("a", ACTION_WEST, "");

		exp.initGUI();


	}


}
