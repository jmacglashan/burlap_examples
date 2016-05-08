package edu.brown.cs.burlap.tutorials.domain.simple;


import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.core.Domain;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.TransitionProbability;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.FullActionModel;
import burlap.mdp.singleagent.GroundedAction;
import burlap.mdp.singleagent.RewardFunction;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.common.SimpleAction;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.explorer.VisualExplorer;
import burlap.mdp.visualizer.StatePainter;
import burlap.mdp.visualizer.StateRenderLayer;
import burlap.mdp.visualizer.Visualizer;

import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.List;

/**
 * @author James MacGlashan.
 */
public class ExampleGridWorld implements DomainGenerator {

	public static final String VAR_X = "x";
	public static final String VAR_Y = "y";

	public static final String ACTION_NORTH = "north";
	public static final String ACTION_SOUTH = "south";
	public static final String ACTION_EAST = "east";
	public static final String ACTION_WEST = "west";


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


	public SADomain generateDomain() {

		SADomain domain = new SADomain();


		new ExampleGridWorld.Movement(ACTION_NORTH, domain, 0);
		new ExampleGridWorld.Movement(ACTION_SOUTH, domain, 1);
		new ExampleGridWorld.Movement(ACTION_EAST, domain, 2);
		new ExampleGridWorld.Movement(ACTION_WEST, domain, 3);


		return domain;
	}


	public StateRenderLayer getStateRenderLayer(){
		StateRenderLayer rl = new StateRenderLayer();
		rl.addStatePainter(new ExampleGridWorld.WallPainter());
		rl.addStatePainter(new ExampleGridWorld.AgentPainter());


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

			//get agent current position
			EXGridState gs = (EXGridState)s;

			int curX = gs.x;
			int curY = gs.y;

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
			gs.x = newPos[0];
			gs.y = newPos[1];

			//return the state we just modified
			return s;
		}


		public List<TransitionProbability> getTransitions(State s, GroundedAction groundedAction) {
			//get agent current position
			EXGridState gs = (EXGridState)s;

			int curX = gs.x;
			int curY = gs.y;

			List<TransitionProbability> tps = new ArrayList<TransitionProbability>(4);
			TransitionProbability noChangeTransition = null;
			for(int i = 0; i < this.directionProbs.length; i++){
				int [] newPos = this.moveResult(curX, curY, i);
				if(newPos[0] != curX || newPos[1] != curY){
					//new possible outcome
					EXGridState ns = gs.copy();
					ns.x = newPos[0];
					ns.y = newPos[1];

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





	public class WallPainter implements StatePainter {

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


	public class AgentPainter implements StatePainter {


		@Override
		public void paint(Graphics2D g2, State s,
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

			int ax = (Integer)s.get(VAR_X);
			int ay = (Integer)s.get(VAR_Y);

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


	public static class ExampleRF implements RewardFunction {

		int goalX;
		int goalY;

		public ExampleRF(int goalX, int goalY){
			this.goalX = goalX;
			this.goalY = goalY;
		}

		public double reward(State s, GroundedAction a, State sprime) {

			int ax = (Integer)s.get(VAR_X);
			int ay = (Integer)s.get(VAR_Y);

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
			int ax = (Integer)s.get(VAR_X);
			int ay = (Integer)s.get(VAR_Y);

			//are they at goal location?
			if(ax == this.goalX && ay == this.goalY){
				return true;
			}

			return false;
		}



	}



	public static void main(String [] args){

		ExampleGridWorld gen = new ExampleGridWorld();
		Domain domain = gen.generateDomain();

		State initialState = new EXGridState(0, 0);

		RewardFunction rf = new ExampleGridWorld.ExampleRF(10, 10);
		TerminalFunction tf = new ExampleGridWorld.ExampleTF(10, 10);

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
