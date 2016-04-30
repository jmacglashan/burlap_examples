package edu.brown.cs.burlap.tutorials;

import burlap.oomdp.auxiliary.DomainGenerator;
import burlap.oomdp.core.*;
import burlap.oomdp.core.objects.MutableObjectInstance;
import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.MutableState;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.FullActionModel;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.SimpleAction;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.visualizer.ObjectPainter;
import burlap.oomdp.visualizer.StateRenderLayer;
import burlap.oomdp.visualizer.StaticPainter;
import burlap.oomdp.visualizer.Visualizer;

import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.List;

/**
 * @author James MacGlashan.
 */
public class ExampleGridWorld implements DomainGenerator{

	public static final String ATT_X = "x";
	public static final String ATT_Y = "y";

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


	public Domain generateDomain() {

		SADomain domain = new SADomain();

		Attribute xatt = new Attribute(domain, ATT_X, Attribute.AttributeType.INT);
		xatt.setLims(0, 10);

		Attribute yatt = new Attribute(domain, ATT_Y, Attribute.AttributeType.INT);
		yatt.setLims(0, 10);

		ObjectClass agentClass = new ObjectClass(domain, CLASS_AGENT);
		agentClass.addAttribute(xatt);
		agentClass.addAttribute(yatt);

		ObjectClass locationClass = new ObjectClass(domain, CLASS_LOCATION);
		locationClass.addAttribute(xatt);
		locationClass.addAttribute(yatt);

		new ExampleGridWorld.Movement(ACTION_NORTH, domain, 0);
		new ExampleGridWorld.Movement(ACTION_SOUTH, domain, 1);
		new ExampleGridWorld.Movement(ACTION_EAST, domain, 2);
		new ExampleGridWorld.Movement(ACTION_WEST, domain, 3);

		new ExampleGridWorld.AtLocation(domain);

		return domain;
	}

	public static State getExampleState(Domain domain){
		State s = new MutableState();
		ObjectInstance agent = new MutableObjectInstance(domain.getObjectClass(CLASS_AGENT), "agent0");
		agent.setValue(ATT_X, 0);
		agent.setValue(ATT_Y, 0);

		ObjectInstance location = new MutableObjectInstance(domain.getObjectClass(CLASS_LOCATION), "location0");
		location.setValue(ATT_X, 10);
		location.setValue(ATT_Y, 10);

		s.addObject(agent);
		s.addObject(location);

		return s;
	}

	public StateRenderLayer getStateRenderLayer(){
		StateRenderLayer rl = new StateRenderLayer();
		rl.addStaticPainter(new ExampleGridWorld.WallPainter());
		rl.addObjectClassPainter(CLASS_LOCATION, new ExampleGridWorld.LocationPainter());
		rl.addObjectClassPainter(CLASS_AGENT, new ExampleGridWorld.AgentPainter());


		return rl;
	}

	public Visualizer getVisualizer(){
		return new Visualizer(this.getStateRenderLayer());
	}


	protected class Movement extends SimpleAction implements FullActionModel {

		//0: north; 1: south; 2:east; 3: west
		protected double [] directionProbs = new double[4];


		public Movement(String actionName, Domain domain, int direction){
			super(actionName, domain);
			for(int i = 0; i < 4; i++){
				if(i == direction){
					directionProbs[i] = 0.8;
				}
				else{
					directionProbs[i] = 0.2/3.;
				}
			}
		}

		@Override
		protected State performActionHelper(State s, GroundedAction groundedAction) {
			//get agent and current position
			ObjectInstance agent = s.getFirstObjectOfClass(CLASS_AGENT);
			int curX = agent.getIntValForAttribute(ATT_X);
			int curY = agent.getIntValForAttribute(ATT_Y);

			//sample directon with random roll
			double r = Math.random();
			double sumProb = 0.;
			int dir = 0;
			for(int i = 0; i < this.directionProbs.length; i++){
				sumProb += this.directionProbs[i];
				if(r < sumProb){
					dir = i;
					break; //found direction
				}
			}

			//get resulting position
			int [] newPos = this.moveResult(curX, curY, dir);

			//set the new position
			agent.setValue(ATT_X, newPos[0]);
			agent.setValue(ATT_Y, newPos[1]);

			//return the state we just modified
			return s;
		}


		public List<TransitionProbability> getTransitions(State s, GroundedAction groundedAction) {
			//get agent and current position
			ObjectInstance agent = s.getFirstObjectOfClass(CLASS_AGENT);
			int curX = agent.getIntValForAttribute(ATT_X);
			int curY = agent.getIntValForAttribute(ATT_Y);

			List<TransitionProbability> tps = new ArrayList<TransitionProbability>(4);
			TransitionProbability noChangeTransition = null;
			for(int i = 0; i < this.directionProbs.length; i++){
				int [] newPos = this.moveResult(curX, curY, i);
				if(newPos[0] != curX || newPos[1] != curY){
					//new possible outcome
					State ns = s.copy();
					ObjectInstance nagent = ns.getFirstObjectOfClass(CLASS_AGENT);
					nagent.setValue(ATT_X, newPos[0]);
					nagent.setValue(ATT_Y, newPos[1]);

					//create transition probability object and add to our list of outcomes
					tps.add(new TransitionProbability(ns, this.directionProbs[i]));
				}
				else{
					//this direction didn't lead anywhere new
					//if there are existing possible directions
					//that wouldn't lead anywhere, aggregate with them
					if(noChangeTransition != null){
						noChangeTransition.p += this.directionProbs[i];
					}
					else{
						//otherwise create this new state and transition
						noChangeTransition = new TransitionProbability(s.copy(),
								this.directionProbs[i]);
						tps.add(noChangeTransition);
					}
				}
			}


			return tps;
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

			int width = ExampleGridWorld.this.map.length;
			int height = ExampleGridWorld.this.map[0].length;

			//make sure new position is valid (not a wall or off bounds)
			if(nx < 0 || nx >= width || ny < 0 || ny >= height ||
					ExampleGridWorld.this.map[nx][ny] == 1){
				nx = curX;
				ny = curY;
			}


			return new int[]{nx,ny};

		}


	}


	protected class AtLocation extends PropositionalFunction {

		public AtLocation(Domain domain){
			super(PF_AT, domain, new String []{CLASS_AGENT, CLASS_LOCATION});
		}

		@Override
		public boolean isTrue(State s, String... params) {
			ObjectInstance agent = s.getObject(params[0]);
			ObjectInstance location = s.getObject(params[1]);

			int ax = agent.getIntValForAttribute(ATT_X);
			int ay = agent.getIntValForAttribute(ATT_Y);

			int lx = location.getIntValForAttribute(ATT_X);
			int ly = location.getIntValForAttribute(ATT_Y);

			return ax == lx && ay == ly;
		}



	}



	public class WallPainter implements StaticPainter {

		public void paint(Graphics2D g2, State s, float cWidth, float cHeight) {

			//walls will be filled in black
			g2.setColor(Color.BLACK);

			//set up floats for the width and height of our domain
			float fWidth = ExampleGridWorld.this.map.length;
			float fHeight = ExampleGridWorld.this.map[0].length;

			//determine the width of a single cell
			//on our canvas such that the whole map can be painted
			float width = cWidth / fWidth;
			float height = cHeight / fHeight;

			//pass through each cell of our map and if it's a wall, paint a black rectangle on our
			//cavas of dimension widthxheight
			for(int i = 0; i < ExampleGridWorld.this.map.length; i++){
				for(int j = 0; j < ExampleGridWorld.this.map[0].length; j++){

					//is there a wall here?
					if(ExampleGridWorld.this.map[i][j] == 1){

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

		public void paintObject(Graphics2D g2, State s, ObjectInstance ob,
								float cWidth, float cHeight) {

			//agent will be filled in gray
			g2.setColor(Color.GRAY);

			//set up floats for the width and height of our domain
			float fWidth = ExampleGridWorld.this.map.length;
			float fHeight = ExampleGridWorld.this.map[0].length;

			//determine the width of a single cell on our canvas
			//such that the whole map can be painted
			float width = cWidth / fWidth;
			float height = cHeight / fHeight;

			int ax = ob.getIntValForAttribute(ATT_X);
			int ay = ob.getIntValForAttribute(ATT_Y);

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

		public void paintObject(Graphics2D g2, State s, ObjectInstance ob,
								float cWidth, float cHeight) {

			//agent will be filled in blue
			g2.setColor(Color.BLUE);

			//set up floats for the width and height of our domain
			float fWidth = ExampleGridWorld.this.map.length;
			float fHeight = ExampleGridWorld.this.map[0].length;

			//determine the width of a single cell on our canvas
			//such that the whole map can be painted
			float width = cWidth / fWidth;
			float height = cHeight / fHeight;

			int ax = ob.getIntValForAttribute(ATT_X);
			int ay = ob.getIntValForAttribute(ATT_Y);

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


	public static class ExampleRF implements RewardFunction {

		int goalX;
		int goalY;

		public ExampleRF(int goalX, int goalY){
			this.goalX = goalX;
			this.goalY = goalY;
		}

		public double reward(State s, GroundedAction a, State sprime) {

			//get location of agent in next state
			ObjectInstance agent = sprime.getFirstObjectOfClass(CLASS_AGENT);
			int ax = agent.getIntValForAttribute(ATT_X);
			int ay = agent.getIntValForAttribute(ATT_Y);

			//are they at goal location?
			if(ax == this.goalX && ay == this.goalY){
				return 100.;
			}

			return -1;
		}


	}

	public static class ExampleTF implements TerminalFunction {

		int goalX;
		int goalY;

		public ExampleTF(int goalX, int goalY){
			this.goalX = goalX;
			this.goalY = goalY;
		}

		public boolean isTerminal(State s) {

			//get location of agent in next state
			ObjectInstance agent = s.getFirstObjectOfClass(CLASS_AGENT);
			int ax = agent.getIntValForAttribute(ATT_X);
			int ay = agent.getIntValForAttribute(ATT_Y);

			//are they at goal location?
			if(ax == this.goalX && ay == this.goalY){
				return true;
			}

			return false;
		}



	}



	public static void main(String [] args){

		burlap.tutorials.bd.ExampleGridWorld gen = new burlap.tutorials.bd.ExampleGridWorld();
		Domain domain = gen.generateDomain();

		State initialState = burlap.tutorials.bd.ExampleGridWorld.getExampleState(domain);

		RewardFunction rf = new burlap.tutorials.bd.ExampleGridWorld.ExampleRF(10, 10);
		TerminalFunction tf = new burlap.tutorials.bd.ExampleGridWorld.ExampleTF(10, 10);

		SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, initialState);

		//TerminalExplorer exp = new TerminalExplorer(domain, env);
		//exp.explore();


		Visualizer v = gen.getVisualizer();
		VisualExplorer exp = new VisualExplorer(domain, env, v);

		exp.addKeyAction("w", ACTION_NORTH);
		exp.addKeyAction("s", ACTION_SOUTH);
		exp.addKeyAction("d", ACTION_EAST);
		exp.addKeyAction("a", ACTION_WEST);

		exp.initGUI();


	}

}
